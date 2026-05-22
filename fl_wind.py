#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



STATIONS = {
    101004: {"name": "Helsinki Kumpula", "file": "Helsinki Kumpula.xlsx"},
    101042: {"name": "Kotka Haapasaari", "file": "Kotka Haapasaari.xlsx"},
    101237: {"name": "Lappeenranta airport", "file": "Lappeenranta airport.xlsx"},
    101150: {"name": "Hämeenlinna Katinen", "file": "Hämeenlinna Katinen.xlsx"},
    101118: {"name": "Pirkkala Tampere-Pirkkala airport", "file": "Pirkkala Tampere-Pirkkala airport.xlsx"},
    100946: {"name": "Hanko Tulliniemi", "file": "Hanko Tulliniemi.xlsx"},
    101065: {"name": "Turku airport", "file": "Turku airport.xlsx"},
    100967: {"name": "Salo Kiikala airfield", "file": "Salo Kiikala airfield.xlsx"},
    101191: {"name": "Kouvola Utti airport", "file": "Kouvola Utti airport.xlsx"},
    855522: {"name": "Mikkeli airport AWOS", "file": "Mikkeli airport AWOS.xlsx"},
    101267: {"name": "Pori Tahkoluoto harbour", "file": "Pori Tahkoluoto harbour.xlsx"},
    151029: {"name": "Mariehamn West Harbour", "file": "Mariehamn West Harbour.xlsx"},
}

FEATURE_COLS = [
    "Wind speed [m/s]",
    "Maximum temperature [°C]",
    "Minimum temperature [°C]",
    "Average relative humidity [%]",
    "Average air pressure [hPa]",
]
TARGET_COL = "target_wind_speed_t_plus_3"


@dataclass
class StationData:
    fmisid: int
    name: str
    raw: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler


def normalize_filename(name: str) -> str:
    return unicodedata.normalize("NFC", name).lower()


def find_file(data_dir: Path, expected_filename: str) -> Path:
    wanted = normalize_filename(expected_filename)
    candidates = list(data_dir.glob("*.xlsx"))
    for path in candidates:
        if normalize_filename(path.name) == wanted:
            return path
    raise FileNotFoundError(
        f"Missing file: {expected_filename}\n"
        f"Looked in: {data_dir.resolve()}\n"
        f"Available Excel files: {[p.name for p in candidates]}"
    )


def load_station_coordinates(data_dir: Path, stations_csv: str) -> None:
    csv_path = data_dir / stations_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing coordinates file: {csv_path}\n"
            "The script needs stations.csv with columns: station_id, station, lat, lon."
        )

    coords = pd.read_csv(csv_path)
    required = {"station_id", "lat", "lon"}
    missing_cols = required - set(coords.columns)
    if missing_cols:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing_cols)}")

    coords["station_id"] = coords["station_id"].astype(int)
    coords = coords.set_index("station_id")

    missing_ids = []
    for sid, info in STATIONS.items():
        if sid not in coords.index:
            missing_ids.append(sid)
        else:
            info["lat"] = float(coords.loc[sid, "lat"])
            info["lon"] = float(coords.loc[sid, "lon"])

    if missing_ids:
        raise ValueError(f"stations.csv is missing these station IDs: {missing_ids}")

    print("Loaded coordinates from", csv_path)
    for sid, info in STATIONS.items():
        print(f"  {sid}: {info['name']} lat={info['lat']:.6f}, lon={info['lon']:.6f}")


def read_station_excel(path: Path, fmisid: int, name: str) -> pd.DataFrame:
    print(f"Reading {fmisid}: {name} from {path.name}")
    df = pd.read_excel(path, engine="openpyxl")

    required_time_cols = ["Year", "Month", "Day", "Time [UTC]"]
    missing_time = [c for c in required_time_cols if c not in df.columns]
    if missing_time:
        raise ValueError(f"{path.name} is missing time columns: {missing_time}")

    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"{path.name} is missing feature columns: {missing_features}")

    date_str = (
        df["Year"].astype(str).str.zfill(4)
        + "-"
        + df["Month"].astype(str).str.zfill(2)
        + "-"
        + df["Day"].astype(str).str.zfill(2)
        + " "
        + df["Time [UTC]"].astype(str)
    )
    df["timestamp"] = pd.to_datetime(date_str, errors="coerce", utc=True)

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df[TARGET_COL] = df["Wind speed [m/s]"].shift(-3)
    df["fmisid"] = fmisid
    df["station_name"] = name

    needed = ["timestamp", "fmisid", "station_name"] + FEATURE_COLS + [TARGET_COL]
    df = df[needed].dropna().reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_start = pd.Timestamp("2025-01-01", tz="UTC")
    val_start = pd.Timestamp("2025-08-01", tz="UTC")
    test_start = pd.Timestamp("2025-10-01", tz="UTC")
    end = pd.Timestamp("2026-01-01", tz="UTC")

    train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < val_start)].copy()
    val = df[(df["timestamp"] >= val_start) & (df["timestamp"] < test_start)].copy()
    test = df[(df["timestamp"] >= test_start) & (df["timestamp"] < end)].copy()

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        print("Warning: date split did not find full 2025 data. Falling back to 60/20/20 chronological split.")
        n = len(df)
        n_train = int(0.60 * n)
        n_val = int(0.20 * n)
        train = df.iloc[:n_train].copy()
        val = df.iloc[n_train:n_train + n_val].copy()
        test = df.iloc[n_train + n_val:].copy()

    return train, val, test


def load_all_data(data_dir: Path) -> Dict[int, StationData]:
    stations: Dict[int, StationData] = {}

    for fmisid, info in STATIONS.items():
        path = find_file(data_dir, info["file"])
        raw = read_station_excel(path, fmisid, info["name"])
        train, val, test = chronological_split(raw)

        scaler = StandardScaler()
        scaler.fit(train[FEATURE_COLS])

        def standardize(split_df: pd.DataFrame) -> pd.DataFrame:
            split_df = split_df.copy()
            scaled = scaler.transform(split_df[FEATURE_COLS])
            for j, col in enumerate(FEATURE_COLS):
                split_df[col] = scaled[:, j].astype(float)
            return split_df

        stations[fmisid] = StationData(
            fmisid=fmisid,
            name=info["name"],
            raw=raw,
            train=standardize(train),
            val=standardize(val),
            test=standardize(test),
            scaler=scaler,
        )

    return stations


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geographic_graph(station_ids: List[int], k: int, sigma_km: float) -> np.ndarray:
    n = len(station_ids)
    A = np.zeros((n, n), dtype=float)

    for i, sid_i in enumerate(station_ids):
        distances = []
        for j, sid_j in enumerate(station_ids):
            if i == j:
                continue
            d = haversine_km(
                STATIONS[sid_i]["lat"],
                STATIONS[sid_i]["lon"],
                STATIONS[sid_j]["lat"],
                STATIONS[sid_j]["lon"],
            )
            distances.append((d, j))

        for d, j in sorted(distances)[:k]:
            weight = math.exp(-d / sigma_km)
            A[i, j] = max(A[i, j], weight)
            A[j, i] = max(A[j, i], weight)

    return A


def correlation_graph(stations: Dict[int, StationData], station_ids: List[int], threshold: float) -> np.ndarray:
    n = len(station_ids)
    A = np.zeros((n, n), dtype=float)

    wind_series = {
        sid: stations[sid].train.set_index("timestamp")["Wind speed [m/s]"]
        for sid in station_ids
    }

    for i, sid_i in enumerate(station_ids):
        for j in range(i + 1, n):
            sid_j = station_ids[j]
            joined = pd.concat([wind_series[sid_i], wind_series[sid_j]], axis=1, join="inner").dropna()
            if len(joined) < 10:
                rho = 0.0
            else:
                rho = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if np.isfinite(rho) and rho >= threshold:
                A[i, j] = max(0.0, rho)
                A[j, i] = max(0.0, rho)

    return A


def graph_edges_dataframe(A: np.ndarray, station_ids: List[int]) -> pd.DataFrame:
    rows = []
    for i in range(len(station_ids)):
        for j in range(i + 1, len(station_ids)):
            if A[i, j] > 0:
                rows.append({
                    "station_i": station_ids[i],
                    "station_j": station_ids[j],
                    "name_i": STATIONS[station_ids[i]]["name"],
                    "name_j": STATIONS[station_ids[j]]["name"],
                    "weight": A[i, j],
                })
    return pd.DataFrame(rows)


def xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)
    return X, y


def add_intercept_column(X: np.ndarray) -> np.ndarray:
    return np.column_stack([X, np.ones(len(X))])


def train_local_closed_form(stations: Dict[int, StationData], station_ids: List[int]) -> np.ndarray:
    params = []
    for sid in station_ids:
        X_train, y_train = xy(stations[sid].train)
        model = LinearRegression()
        model.fit(X_train, y_train)
        params.append(np.r_[model.coef_, model.intercept_])
    return np.vstack(params)


def train_gtvmin(
    stations: Dict[int, StationData],
    station_ids: List[int],
    A: np.ndarray,
    alpha: float,
    iterations: int,
    lr: float,
    regularize_intercept: bool,
) -> np.ndarray:
    n = len(station_ids)
    d = len(FEATURE_COLS) + 1
    W = np.zeros((n, d), dtype=float)

    train_arrays = []
    for sid in station_ids:
        X, y = xy(stations[sid].train)
        train_arrays.append((add_intercept_column(X), y))

    reg_mask = np.ones(d, dtype=float)
    if not regularize_intercept:
        reg_mask[-1] = 0.0

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

        if not np.all(np.isfinite(W)):
            raise FloatingPointError(
                f"GTVMin diverged at iteration {it}. Try smaller --lr, e.g. --lr 0.001"
            )

    return W


def evaluate_params(W: np.ndarray, stations: Dict[int, StationData], station_ids: List[int], split: str):
    rows = []
    for i, sid in enumerate(station_ids):
        df = getattr(stations[sid], split)
        X, y = xy(df)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Folder containing station Excel files and stations.csv")
    parser.add_argument("--stations_csv", type=str, default="stations.csv", help="CSV with station_id, lat, lon")
    parser.add_argument("--out_dir", type=str, default="results", help="Output folder")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--sigma_km", type=float, default=50.0)
    parser.add_argument("--regularize_intercept", action="store_true", help="Also graph-regularize intercepts")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument("--corr_thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_station_coordinates(data_dir, args.stations_csv)
    stations = load_all_data(data_dir)
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
    A_geo = geographic_graph(station_ids, k=args.k, sigma_km=args.sigma_km)
    graph_edges_dataframe(A_geo, station_ids).to_csv(out_dir / "system_A_geographic_edges.csv", index=False)

    print("Selecting alpha for System A using validation MSE...")
    best_geo = None
    for alpha in args.alphas:
        try:
            W_geo = train_gtvmin(stations, station_ids, A_geo, alpha, args.iterations, args.lr, args.regularize_intercept)
            val_mse, val_mae, _ = evaluate_params(W_geo, stations, station_ids, "val")
            print(f"  System A alpha={alpha:g}: val MSE={val_mse:.4f}, val MAE={val_mae:.4f}")
        except FloatingPointError as exc:
            print(f"  System A alpha={alpha:g}: skipped ({exc})")
            continue
        if best_geo is None or val_mse < best_geo["val_mse"]:
            best_geo = {"alpha": alpha, "val_mse": val_mse, "W": W_geo}

    if best_geo is None:
        raise RuntimeError("All System A alpha values diverged. Try --lr 0.001")

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
                W_corr = train_gtvmin(stations, station_ids, A_corr, alpha, args.iterations, args.lr, args.regularize_intercept)
                val_mse, val_mae, _ = evaluate_params(W_corr, stations, station_ids, "val")
                print(f"  System B threshold={threshold:g}, alpha={alpha:g}, edges={n_edges}: val MSE={val_mse:.4f}, val MAE={val_mae:.4f}")
            except FloatingPointError as exc:
                print(f"  System B threshold={threshold:g}, alpha={alpha:g}: skipped ({exc})")
                continue
            if best_corr is None or val_mse < best_corr["val_mse"]:
                best_corr = {"threshold": threshold, "alpha": alpha, "val_mse": val_mse, "W": W_corr, "A": A_corr, "n_edges": n_edges}

    if best_corr is None:
        raise RuntimeError("No valid correlation graph. Try lower thresholds, e.g. --corr_thresholds 0.1 0.2 0.3 0.4 0.5")

    graph_edges_dataframe(best_corr["A"], station_ids).to_csv(out_dir / "system_B_correlation_edges.csv", index=False)
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
    
    #visualization
    # Station-level test MSE figure

    import matplotlib.pyplot as plt

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
        
    visualize_graph(
        A_geo,
        station_ids,
        "System A Geographic Graph",
        out_dir / "system_A_graph.png"
    )

    visualize_graph(
        best_corr["A"],
        station_ids,
        "System B Correlation Graph",
        out_dir / "system_B_graph.png"
    )


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
