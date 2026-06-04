#!/usr/bin/env python3
from __future__ import annotations

from data_processing import *
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# add or remove stations for FL network 
STATIONS = {
    100946: {"name": "Hanko Tulliniemi", "file": "100946_hanko_tulliniemi_hourly_2025.csv"},
    100967: {"name": "Salo Kiikala airfield", "file": "100967_salo_kiikala_airfield_hourly_2025.csv"},
    101004: {"name": "Helsinki Kumpula", "file": "101004_helsinki_kumpula_hourly_2025.csv"},
    101022: {"name": "Porvoo Kalbådagrund", "file": "101022_porvoo_kalbådagrund_hourly_2025.csv"},
    101042: {"name": "Kotka Haapasaari", "file": "101042_kotka_haapasaari_hourly_2025.csv"},
    101065: {"name": "Turku airport", "file": "101065_turku_airport_hourly_2025.csv"},
    101118: {"name": "Pirkkala Tampere-Pirkkala airport", "file": "101118_pirkkala_tampere_pirkkala_airport_hourly_2025.csv"},
    101150: {"name": "Hämeenlinna Katinen", "file": "101150_hameenlinna_katinen_hourly_2025.csv"},
    101191: {"name": "Kouvola Utti airport", "file": "101191_kouvola_utti_airport_hourly_2025.csv"},
    101237: {"name": "Lappeenranta airport", "file": "101237_lappeenranta_airport_hourly_2025.csv"},
    101267: {"name": "Pori Tahkoluoto harbour", "file": "101267_pori_tahkoluoto_harbour_hourly_2025.csv"},
    #101783: {"name": "Kemi I majakka", "file": "101783_kemi_i_majakka_hourly_2025.csv"},
    #101794: {"name": "Oulu Vihreäsaari satama", "file": "101794_oulu_vihreasaari_satama_hourly_2025.csv"},
    #101851: {"name": "Tornio Kaakkuri", "file": "101851_tornio_kaakkuri_hourly_2025.csv"},
    #101928: {"name": "Rovaniemi rautatieasema", "file": "101928_rovaniemi_rautatieasema_hourly_2025.csv"},
    #101950: {"name": "Kemijärvi lentokenttä", "file": "101950_kemijarvi_lentokentta_hourly_2025.csv"},
    #102033: {"name": "Inari Ivalo lentoasema", "file": "102033_inari_ivalo_lentoasema_hourly_2025.csv"},
    #106435: {"name": "Muonio Oustajärvi", "file": "106435_muonio_oustajarvi_hourly_2025.csv"},
    151029: {"name": "Mariehamn West Harbour", "file": "151029_mariehamn_west_harbour_hourly_2025.csv"},
    855522: {"name": "Mikkeli airport AWOS", "file": "855522_mikkeli_airport_awos_hourly_2025.csv"},
    874863: {"name": "Espoo Tapiola", "file": "874863_espoo_tapiola_hourly_2025.csv"},
}

FEATURE_COLS = [
    "Wind speed [m/s]",
    "Maximum temperature [°C]",
    "Minimum temperature [°C]",
    "Average relative humidity [%]",
    "Average air pressure [hPa]",
]
TARGET_COL = "target_wind_speed_t_plus_3"


def train_local_closed_form(stations: Dict[int, StationData], station_ids: List[int]) -> np.ndarray:
    params = []
    for sid in station_ids:
        X_train, y_train = xy(stations[sid].train, FEATURE_COLS, TARGET_COL)
        model = LinearRegression()
        model.fit(X_train, y_train)
        params.append(np.r_[model.coef_, model.intercept_])
    return np.vstack(params)


def train_gtvmin(stations: Dict[int, StationData], station_ids: List[int], A: np.ndarray, 
                 alpha: float, iterations: int, lr: float, regularize_intercept: bool, log_every: int = 50,
                 ) -> Tuple[np.ndarray, List]:
    n = len(station_ids)
    d = len(FEATURE_COLS) + 1
    W = np.zeros((n, d), dtype=float)

    train_arrays = []
    for sid in station_ids:
        X, y = xy(stations[sid].train, FEATURE_COLS, TARGET_COL)
        train_arrays.append((add_intercept_column(X), y))

    reg_mask = np.ones(d, dtype=float)
    if not regularize_intercept:
        reg_mask[-1] = 0.0
    loss_history = []

    for it in range(iterations):
        W_old = W.copy()
        grad = np.zeros_like(W)

        for i, (X_i, y_i) in enumerate(train_arrays):
            m_i = len(y_i)
            residual = X_i @ W_old[i] - y_i
            grad_loss = (2.0 / m_i) * (X_i.T @ residual)

            diff = W_old[i] - W_old
            grad_reg = 2.0 * alpha * (A[i, :, None] * diff).sum(axis=0)
            grad_reg = grad_reg * reg_mask

            grad[i] = grad_loss + grad_reg

        W = W_old - lr * grad

        if it % log_every == 0:
            total_loss = 0.0
            for i, (X_i, y_i) in enumerate(train_arrays):
                total_loss += float(np.mean((X_i @ W[i] - y_i) ** 2))
                for j in range(n):
                    if A[i, j] > 0:
                        diff_ij = (W[i] - W[j]) * reg_mask
                        total_loss += alpha * A[i, j] * float(np.dot(diff_ij, diff_ij))
            loss_history.append((it, total_loss))

        if not np.all(np.isfinite(W)):
            raise FloatingPointError(
                f"GTVMin diverged at iteration {it}. Try smaller --lr, e.g. --lr 0.001"
            )

    return W, loss_history


def evaluate_params(W: np.ndarray, stations: Dict[int, StationData], station_ids: List[int], split: str):
    rows = []
    for i, sid in enumerate(station_ids):
        df = getattr(stations[sid], split)
        X, y = xy(df, FEATURE_COLS, TARGET_COL)
        pred = add_intercept_column(X) @ W[i]
        rows.append({
            "fmisid": sid,
            "station": STATIONS[sid]["name"],
            "split": split,
            "mse": mean_squared_error(y, pred),
            "mae": mean_absolute_error(y, pred),
            "n_samples": len(y),
        })
    metrics = pd.DataFrame(rows)
    return float(metrics["mse"].mean()), float(metrics["mae"].mean()), metrics


def evaluate_all_splits(method_name: str, W: np.ndarray, stations: Dict[int, StationData], station_ids: List[int]):
    summary = {"method": method_name}
    details = []
    for split in ["train", "val", "test"]:
        mse, mae, detail = evaluate_params(W, stations, station_ids, split)
        summary[f"{split}_mse"] = mse
        summary[f"{split}_mae"] = mae
        detail["method"] = method_name
        details.append(detail)
    return summary, pd.concat(details, ignore_index=True)


def sample_count_table(stations: Dict[int, StationData], station_ids: List[int]) -> pd.DataFrame:
    rows = []
    for sid in station_ids:
        rows.append({
            "fmisid": sid,
            "station": STATIONS[sid]["name"],
            "train": len(stations[sid].train),
            "validation": len(stations[sid].val),
            "test": len(stations[sid].test),
            "total": len(stations[sid].raw),
        })
    return pd.DataFrame(rows)

def visualize_graph(A, station_ids, title, save_path):
    G = nx.Graph()

    # add nodes
    for sid in station_ids:
        G.add_node(sid)

    # add weighted edges
    for i in range(len(station_ids)):
        for j in range(i + 1, len(station_ids)):
            if A[i, j] > 0:
                G.add_edge(
                    station_ids[i],
                    station_ids[j],
                    weight=round(A[i, j], 2)
                )

    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.5
    )

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1800,
        font_size=10,
        width=2
    )

    edge_labels = nx.get_edge_attributes(G, "weight")

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False
    )

    plt.title(title)
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/fmi_hourly_2025", help="Folder containing station csv files and stations.csv")
    parser.add_argument("--stations_csv", type=str, default="../stations.csv", help="CSV with station_id, lat, lon")
    parser.add_argument("--out_dir", type=str, default="results", help="Output folder")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--sigma_km", type=float, default=50.0)
    parser.add_argument("--regularize_intercept", action="store_true", help="Also graph-regularize intercepts")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0001, 0.0005, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05])
    parser.add_argument("--corr_thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_station_coordinates(data_dir, args.stations_csv, STATIONS)
    stations = load_all_data(data_dir, STATIONS, FEATURE_COLS, TARGET_COL)
    station_ids = list(STATIONS.keys())

    counts = sample_count_table(stations, station_ids)
    counts.to_csv(out_dir / "sample_counts.csv", index=False)
    print("\nSample counts:")
    print(counts.to_string(index=False))

    summaries = []
    details = []

    print("\nTraining local-only baseline...")
    W_local = train_local_closed_form(stations, station_ids)
    summary, detail = evaluate_all_splits("Local only (alpha=0)", W_local, stations, station_ids)
    summaries.append(summary)
    details.append(detail)

    print("\nConstructing System A geographic graph...")
    A_geo = geographic_graph(station_ids, k=args.k, sigma_km=args.sigma_km, stations=STATIONS)
    graph_edges_dataframe(A_geo, station_ids, STATIONS).to_csv(out_dir / "system_A_geographic_edges.csv", index=False)

    print("Selecting alpha for System A using validation MSE...")
    best_geo = None
    for alpha in args.alphas:
        try:
            W_geo, _ = train_gtvmin(stations, station_ids, A_geo, alpha, args.iterations, args.lr, args.regularize_intercept)
            val_mse, val_mae, _ = evaluate_params(W_geo, stations, station_ids, "val")
            print(f"  System A alpha={alpha:g}: val MSE={val_mse:.4f}, val MAE={val_mae:.4f}")
        except FloatingPointError as exc:
            print(f"  System A alpha={alpha:g}: skipped ({exc})")
            continue
        if best_geo is None or val_mse < best_geo["val_mse"]:
            best_geo = {"alpha": alpha, "val_mse": val_mse, "W": W_geo}

    if best_geo is None:
        raise RuntimeError("All System A alpha values diverged. Try --lr 0.001")
        
    _, loss_hist_geo = train_gtvmin(
        stations, station_ids, A_geo,
        best_geo["alpha"], args.iterations, args.lr, args.regularize_intercept
    )

    summary, detail = evaluate_all_splits(
        f"System A Geographic (alpha={best_geo['alpha']:g}, k={args.k})",
        best_geo["W"], stations, station_ids
    )
    summaries.append(summary)
    details.append(detail)

    print("\nSelecting threshold and alpha for System B using validation MSE...")
    best_corr = None
    for threshold in args.corr_thresholds:
        A_corr = correlation_graph(stations, station_ids, threshold)
        n_edges = int((A_corr > 0).sum() // 2)
        if n_edges == 0:
            print(f"  threshold={threshold:g}: skipped because graph has no edges")
            continue
        for alpha in args.alphas:
            try:
                W_corr, _ = train_gtvmin(stations, station_ids, A_corr, alpha, args.iterations, args.lr, args.regularize_intercept)
                val_mse, val_mae, _ = evaluate_params(W_corr, stations, station_ids, "val")
                print(f"  System B threshold={threshold:g}, alpha={alpha:g}, edges={n_edges}: val MSE={val_mse:.4f}, val MAE={val_mae:.4f}")
            except FloatingPointError as exc:
                print(f"  System B threshold={threshold:g}, alpha={alpha:g}: skipped ({exc})")
                continue
            if best_corr is None or val_mse < best_corr["val_mse"]:
                best_corr = {"threshold": threshold, "alpha": alpha, "val_mse": val_mse, "W": W_corr, "A": A_corr, "n_edges": n_edges}

    if best_corr is None:
        raise RuntimeError("No valid correlation graph. Try lower thresholds, e.g. --corr_thresholds 0.1 0.2 0.3 0.4 0.5")

    _, loss_hist_corr = train_gtvmin(
        stations, station_ids, best_corr["A"],
        best_corr["alpha"], args.iterations, args.lr, args.regularize_intercept
    )

    graph_edges_dataframe(best_corr["A"], station_ids, STATIONS).to_csv(out_dir / "system_B_correlation_edges.csv", index=False)
    summary, detail = evaluate_all_splits(
        f"System B Correlation (alpha={best_corr['alpha']:g}, rho={best_corr['threshold']:g})",
        best_corr["W"], stations, station_ids
    )
    summaries.append(summary)
    details.append(detail)

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True)
    summary_df.to_csv(out_dir / "summary_results.csv", index=False)
    detail_df.to_csv(out_dir / "station_level_results.csv", index=False)

    param_cols = [f"coef_{c}" for c in FEATURE_COLS] + ["intercept"]
    param_rows = []
    for method_name, W in [
        ("Local only", W_local),
        (f"System A Geographic alpha={best_geo['alpha']:g}", best_geo["W"]),
        (f"System B Correlation alpha={best_corr['alpha']:g} rho={best_corr['threshold']:g}", best_corr["W"]),
    ]:
        for i, sid in enumerate(station_ids):
            row = {"method": method_name, "fmisid": sid, "station": STATIONS[sid]["name"]}
            row.update({col: W[i, j] for j, col in enumerate(param_cols)})
            param_rows.append(row)
    pd.DataFrame(param_rows).to_csv(out_dir / "learned_parameters.csv", index=False)
    

    iters_A, losses_A = zip(*loss_hist_geo)
    iters_B, losses_B = zip(*loss_hist_corr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(iters_A, losses_A)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("GTVMin objective")
    axes[0].set_title(f"System A convergence (alpha={best_geo['alpha']:g})")
    axes[0].set_yscale("log")

    axes[1].plot(iters_B, losses_B)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("GTVMin objective")
    axes[1].set_title(f"System B convergence (alpha={best_corr['alpha']:g}, rho={best_corr['threshold']:g})")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_dir / "convergence_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    #visualization
    # Station-level test MSE figure

    test_detail = detail_df[detail_df["split"] == "test"]

    pivot = test_detail.pivot(
        index="station",
        columns="method",
        values="mse"
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    pivot.plot(kind="bar", ax=ax)

    ax.set_ylabel("Test MSE")
    ax.set_title("Station-level Test MSE")

    plt.xticks(rotation=90, ha="center")

    # move legend outside
    ax.legend(loc= "upper center")

    plt.tight_layout()

    plt.savefig(
        out_dir / "station_test_mse.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
        
    visualize_graph(A_geo, station_ids, "System A Geographic Graph", out_dir / "system_A_graph.png")

    visualize_graph(best_corr["A"], station_ids, "System B Correlation Graph", out_dir / "system_B_graph.png")


    print("\nFinal average results:")
    print(summary_df.to_string(index=False))
    print(f"\nBest System A alpha: {best_geo['alpha']}")
    print(f"Best System B alpha: {best_corr['alpha']}, threshold: {best_corr['threshold']}, edges: {best_corr['n_edges']}")
    print(f"\nSaved files in: {out_dir.resolve()}")
    print("  sample_counts.csv")
    print("  summary_results.csv")
    print("  station_level_results.csv")
    print("  learned_parameters.csv")
    print("  system_A_geographic_edges.csv")
    print("  system_B_correlation_edges.csv")


if __name__ == "__main__":
    main()
