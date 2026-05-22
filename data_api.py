import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone


BASE_URL = "https://opendata.fmi.fi/wfs"

STORED_QUERY = "fmi::observations::weather::hourly::simple"

PARAMETERS = [
    "TA_PT1H_MAX",  # Maximum temperature [°C]
    "TA_PT1H_MIN",  # Minimum temperature [°C]
    "RH_PT1H_AVG",  # Average relative humidity [%]
    "WS_PT1H_AVG",  # Wind speed [m/s]
    "PA_PT1H_AVG",  # Average air pressure [hPa]
]

STATIONS = {
    101004: "Helsinki Kumpula",
    104796: "Lahti Sopenkorpi",
    101042: "Kotka Haapasaari",
    101237: "Lappeenranta airport",
    101150: "Hämeenlinna Katinen",
    101118: "Pirkkala Tampere-Pirkkala airport",
    100946: "Hanko Tulliniemi",
    101065: "Turku airport",
    100967: "Salo Kiikala airfield",
    101191: "Kouvola Utti airport",
    855522: "Mikkeli airport AWOS",
    101267: "Pori Tahkoluoto harbour",
    151029: "Mariehamn West Harbour", 
}

OUT_DIR = Path("data/fmi_hourly_2025")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def local_name(tag):
    return tag.split("}", 1)[-1]


def month_ranges(year):
    for month in range(1, 13):
        start = datetime(year, month, 1, 0, 0, tzinfo=timezone.utc)

        if month == 12:
            end = datetime(year + 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        else:
            end = datetime(year, month + 1, 1, 0, 0, tzinfo=timezone.utc)

        yield start, end


def fetch_one_month(fmisid, start, end):
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "getFeature",
        "storedquery_id": STORED_QUERY,
        "fmisid": str(fmisid),
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timestep": "60",
        "parameters": ",".join(PARAMETERS),
    }

    response = requests.get(BASE_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.text


def parse_fmi_simple_xml(xml_text, fmisid, station_name):
    root = ET.fromstring(xml_text)

    rows = []

    for elem in root.iter():
        if local_name(elem.tag) != "BsWfsElement":
            continue

        record = {
            "fmisid": fmisid,
            "station": station_name,
        }

        for child in elem:
            name = local_name(child.tag)
            text = child.text.strip() if child.text else None
            record[name] = text

        rows.append(record)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    required_columns = {"Time", "ParameterName", "ParameterValue"}
    if not required_columns.issubset(df.columns):
        print("Unexpected columns:", df.columns.tolist())
        return pd.DataFrame()

    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df["ParameterValue"] = pd.to_numeric(df["ParameterValue"], errors="coerce")

    wide = (
        df.pivot_table(
            index=["fmisid", "station", "Time"],
            columns="ParameterName",
            values="ParameterValue",
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns.name = None
    return wide


def download_station(fmisid, station_name, year=2025):
    print(f"\nDownloading {station_name} ({fmisid})")

    monthly_frames = []

    for start, end in month_ranges(year):
        print(f"  {start:%Y-%m} ...", end=" ")

        try:
            xml_text = fetch_one_month(fmisid, start, end)
            df_month = parse_fmi_simple_xml(xml_text, fmisid, station_name)

            if not df_month.empty:
                monthly_frames.append(df_month)
                print(f"{len(df_month)} hourly rows")
            else:
                print("no data")

        except Exception as e:
            print(f"failed: {e}")

        time.sleep(0.5)

    if not monthly_frames:
        return pd.DataFrame()

    df = pd.concat(monthly_frames, ignore_index=True)
    df = df.drop_duplicates(subset=["fmisid", "Time"])
    df = df.sort_values("Time")

    return df


def clean_columns(df):
    rename_map = {
        "Time": "time_utc",
        "TA_PT1H_MAX": "Maximum temperature [°C]",
        "TA_PT1H_MIN": "Minimum temperature [°C]",
        "RH_PT1H_AVG": "Average relative humidity [%]",
        "WS_PT1H_AVG": "Wind speed [m/s]",
        "PA_PT1H_AVG": "Average air pressure [hPa]",
    }

    df = df.rename(columns=rename_map)

    keep_columns = [
        "fmisid",
        "station",
        "time_utc",
        "Maximum temperature [°C]",
        "Minimum temperature [°C]",
        "Average relative humidity [%]",
        "Wind speed [m/s]",
        "Average air pressure [hPa]",
    ]

    existing_columns = [col for col in keep_columns if col in df.columns]
    return df[existing_columns]


def make_safe_filename(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("ä", "a")
        .replace("ö", "o")
        .replace("/", "_")
    )


def main():
    all_frames = []

    for fmisid, station_name in STATIONS.items():
        df_station = download_station(fmisid, station_name, year=2025)

        if df_station.empty:
            print(f"No data saved for {station_name}")
            continue

        df_station = clean_columns(df_station)

        safe_name = make_safe_filename(station_name)
        out_file = OUT_DIR / f"{fmisid}_{safe_name}_hourly_2025.csv"

        df_station.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

        all_frames.append(df_station)

    if all_frames:
        df_all = pd.concat(all_frames, ignore_index=True)
        df_all = df_all.sort_values(["station", "time_utc"])

        combined_file = OUT_DIR / "fmi_12_stations_hourly_2025.csv"
        df_all.to_csv(combined_file, index=False)

        print("\nDone.")
        print(f"Combined file: {combined_file}")
        print(f"Rows: {len(df_all)}")
        print("Columns:")
        print(df_all.columns.tolist())
    else:
        print("No data downloaded.")


if __name__ == "__main__":
    main()