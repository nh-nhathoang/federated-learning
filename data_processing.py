
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    candidates = list(data_dir.glob("*.csv"))
    for path in candidates:
        if normalize_filename(path.name) == wanted:
            return path
    raise FileNotFoundError(
        f"Missing file: {expected_filename}\n"
        f"Looked in: {data_dir.resolve()}\n"
        f"Available csv files: {[p.name for p in candidates]}"
    )

def load_station_coordinates(data_dir: Path, stations_csv: str, stations: Dict[int, Dict]) -> None:
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
    for sid, info in stations.items():
        if sid not in coords.index:
            missing_ids.append(sid)
        else:
            info["lat"] = float(coords.loc[sid, "lat"])
            info["lon"] = float(coords.loc[sid, "lon"])

    if missing_ids:
        raise ValueError(f"stations.csv is missing these station IDs: {missing_ids}")

    print("Loaded coordinates from", csv_path)
    for sid, info in stations.items():
        print(f"  {sid}: {info['name']} lat={info['lat']:.6f}, lon={info['lon']:.6f}")

def read_station_csv(path: Path, fmisid: int, name: str, features: List[str], target_col: str) -> pd.DataFrame:
    print(f"Reading {fmisid}: {name} from {path.name}")
    df = pd.read_csv(path)

    if "time_utc" not in df.columns:
        raise ValueError(f"{path.name} is missing column: time_utc")
    
    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        raise ValueError(f"{path.name} is missing feature columns: {missing_features}")

    df["timestamp"] = pd.to_datetime(df["time_utc"], errors="coerce", utc=True)

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df[target_col] = df["Wind speed [m/s]"].shift(-3)
    df["fmisid"] = fmisid
    df["station_name"] = name

    needed = ["timestamp", "fmisid", "station_name"] + features + [target_col]
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

def load_all_data(data_dir: Path, stations: Dict[int, Dict], features: List[str], target_col: str) -> Dict[int, StationData]:
    result: Dict[int, StationData] = {}

    for fmisid, info in stations.items():
        path = find_file(data_dir, info["file"])
        raw = read_station_csv(path, fmisid, info["name"], features, target_col)
        train, val, test = chronological_split(raw)

        scaler = StandardScaler()
        scaler.fit(train[features])

        def standardize(split_df: pd.DataFrame) -> pd.DataFrame:
            split_df = split_df.copy()
            scaled = scaler.transform(split_df[features])
            for j, col in enumerate(features):
                split_df[col] = scaled[:, j].astype(float)
            return split_df

        result[fmisid] = StationData(
            fmisid=fmisid,
            name=info["name"],
            raw=raw,
            train=standardize(train),
            val=standardize(val),
            test=standardize(test),
            scaler=scaler,
        )

    return result
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geographic_graph(station_ids: List[int], k: int, sigma_km: float, stations: Dict[int, Dict]) -> np.ndarray:
    n = len(station_ids)
    A = np.zeros((n, n), dtype=float)

    for i, sid_i in enumerate(station_ids):
        distances = []
        for j, sid_j in enumerate(station_ids):
            if i == j:
                continue
            d = haversine_km(
                stations[sid_i]["lat"],
                stations[sid_i]["lon"],
                stations[sid_j]["lat"],
                stations[sid_j]["lon"],
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


def graph_edges_dataframe(A: np.ndarray, station_ids: List[int], stations: Dict[int, Dict]) -> pd.DataFrame:
    rows = []
    for i in range(len(station_ids)):
        for j in range(i + 1, len(station_ids)):
            if A[i, j] > 0:
                rows.append({
                    "station_i": station_ids[i],
                    "station_j": station_ids[j],
                    "name_i": stations[station_ids[i]]["name"],
                    "name_j": stations[station_ids[j]]["name"],
                    "weight": A[i, j],
                })
    return pd.DataFrame(rows)

def xy(df: pd.DataFrame, features: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    X = df[features].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return X, y


def add_intercept_column(X: np.ndarray) -> np.ndarray:
    return np.column_stack([X, np.ones(len(X))])
