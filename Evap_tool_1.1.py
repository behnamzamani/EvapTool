# EvapTool ver. 1.1
# Copyright (C) 2026 Behnam Zamani
#
# This program is licensed under the GNU Affero General Public License v3.0.
# See the LICENSE file for details.


"""
version log:

Version 1.1
downloads recent daily meteorological data from closest DWD weather stations to desired location. 
Data are downloaded automatically from DWD FTP server.
daily mean evaporation is calculated by Penman and Priestley-Taylor approaches.
water balance calculated as difference between precipitation and evaporation.

"""
import sys
import numpy as np
import pandas as pd
import math
import datetime
import os
from pandas import Timestamp
import time
import glob
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from scipy import signal
import pymannkendall as mk
import multiprocessing as mp
import re
from time import sleep
import itertools
from tqdm import tqdm
import random
import fileinput
import urllib.request
from progress.bar import Bar
import subprocess
import shutil
import netCDF4 as nc
import matplotlib
#matplotlib.use('Agg') #-- active when on cluster
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from pandas import Timestamp
import matplotlib.dates as dates
import pyny3d.geoms as pyny
import warnings
import random
import decimal
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
import scipy
import sys
import zipfile
import shutil

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

version=os.path.splitext(os.path.basename(__file__))[0]


######################### download weather data from DWD website ####################################
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
import requests
import re

# =====================================================
# SCREEN DISPLAY FUNCTIONS
# =====================================================

HEADER_TEXT = (
    "***************************** EvapTool ver 1.1 *******************************\n"
    "This program downloads recent daily meteorological data from closest\n"
    "DWD weather stations to desired location.\n\n"
    "Data are downloaded automatically from DWD FTP server\n"
    "Daily evaporation is calculated by Penman and Priestley-Taylor approaches.\n\n"
    "water balance calculated as difference between precipitation and evaporation.\n\n"
    "by: Behnam Zamani\n"
)

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def show_header():
    print("\n" + "=" * 78)
    print(" " * 30 + "EvapTool ver. 1.1")
    print("=" * 78)
    print("\n" + HEADER_TEXT + "\n")
    print("=" * 78)


def show_status(message):
    clear_screen()
    show_header()
    print("\nStatus:")
    print("-" * 78)
    print(message)
    print("-" * 78)


# =====================================================
# SETTINGS
# =====================================================

BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/historical/"
RECENT_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/daily/kl/recent/"
)

STATION_FILE = "KL_Tageswerte_Beschreibung_Stationen.txt"

download_dir = Path("data_download")
download_dir.mkdir(exist_ok=True)


# =====================================================
# USER INPUT / BENUTZEREINGABE
# =====================================================

coord_str = input(
    "Enter coordinates as Latitude (Hochwert),Longitude (Rechtswert)\n"
    "(e.g. 52.3759,9.7320) WGS84: "
).strip()

#coord_str='51.4761272,6.3996615'


try:

    lat, lon = [
        float(x.strip())
        for x in coord_str.split(",")
    ]

except Exception:

    show_status(
        "\nInvalid coordinate format."
        "\nExample: 52.3759,9.7320"
    )

    sys.exit()


# =====================================================
# READ RECENT DWD STATION LIST
# =====================================================

show_status("\nLoading recent station information...")

station_file_local = download_dir / STATION_FILE

# Use RECENT_URL because we want current/recent station availability
urlretrieve(
    RECENT_URL + STATION_FILE,
    station_file_local
)

stations = []

with open(station_file_local, "r", encoding="latin1") as f:

    for line in f:

        if line.startswith("Stations_id"):
            continue

        if line.startswith("-"):
            continue

        if not line.strip():
            continue

        parts = line.strip().split()

        if len(parts) < 8:
            continue

        try:

            station_id = parts[0].zfill(5)

            von_datum = parts[1]
            bis_datum = parts[2]

            stationshoehe = float(parts[3])

            geo_breite = float(parts[4])
            geo_laenge = float(parts[5])

            station_name = " ".join(parts[6:-2])

            stations.append(
                {
                    "Stations_id": station_id,
                    "Stationsname": station_name,
                    "Stationshoehe": stationshoehe,
                    "geoBreite": geo_breite,
                    "geoLaenge": geo_laenge,
                    "von_datum": von_datum,
                    "bis_datum": bis_datum
                }
            )

        except Exception:
            continue


stations_df = pd.DataFrame(stations)


# =====================================================
# PREPARE bis_datum AND DISTANCE
# =====================================================

stations_df["bis_datum"] = pd.to_datetime(
    stations_df["bis_datum"],
    format="%Y%m%d",
    errors="coerce"
)

stations_df["distance"] = np.sqrt(
    (stations_df["geoBreite"] - lat) ** 2
    + (stations_df["geoLaenge"] - lon) ** 2
)

stations_df = (
    stations_df
    .dropna(subset=["bis_datum"])
    .sort_values("distance")
    .reset_index(drop=True)
)


# =====================================================
# FIND CLOSEST STATION WITH RECENT DATA
# First try: data until at least 1 month ago
# If none found: data until at least 2 months ago
# =====================================================

station_code = None
station_name = None
closest_station = None
used_recency_months = None

for months_back in [1, 2]:

    cutoff_date = (
        pd.Timestamp.today().normalize()
        - pd.DateOffset(months=months_back)
    )

    show_status(
        f"\nSearching closest DWD climate station with data "
        f"until at least {cutoff_date.date()}..."
    )

    candidate_df = stations_df[
        stations_df["bis_datum"] >= cutoff_date
    ].copy()

    if len(candidate_df) == 0:

        show_status(
            f"No station found with data until "
            f"{cutoff_date.date()}."
        )

        continue

    closest_station = candidate_df.iloc[0]

    station_code = closest_station["Stations_id"]
    station_name = closest_station["Stationsname"]
    used_recency_months = months_back

    station_code_final=station_code
    station_name_final=station_name



    # Extract station elevation and coordinates
    elev_st = float(closest_station["Stationshoehe"])
    lat = float(closest_station["geoBreite"])
    lon = float(closest_station["geoLaenge"])


    print(
        f"\nClosest DWD climate station with recent data:"
        f"\nStation name : {station_name}"
        f"\nStation code : {station_code}"
        f"\nLast data    : {closest_station['bis_datum'].date()}"
        f"\nCriterion    : within last {months_back} month(s)"
    )

    break


# =====================================================
# STOP IF NO SUITABLE STATION WAS FOUND
# =====================================================

if station_code is None:

    raise RuntimeError(
        "No suitable DWD climate station found with data "
        "within the last 1 or 2 months."
    )


# =====================================================
# DOWNLOAD RECENT DATA FOR SELECTED STATION
# =====================================================

show_status("\nDownloading recent climate data for selected station...")

recent_zip_filename = f"tageswerte_KL_{station_code}_akt.zip"
recent_zip_url = RECENT_URL + recent_zip_filename

recent_zip_file = download_dir / recent_zip_filename

show_status(f"Downloading:\n{recent_zip_filename}")

urlretrieve(
    recent_zip_url,
    recent_zip_file
)

show_status("Recent data download completed.")


# =====================================================
# EXTRACT RECENT ZIP
# =====================================================

recent_extract_dir = download_dir / f"klima_recent_{station_code}"

recent_extract_dir.mkdir(
    parents=True,
    exist_ok=True
)

show_status("\nExtracting recent ZIP file...")

with zipfile.ZipFile(recent_zip_file, "r") as zip_ref:
    zip_ref.extractall(recent_extract_dir)

show_status("Extraction completed.")


# =====================================================
# FIND RECENT KLIMA FILE
# =====================================================

recent_files = list(
    recent_extract_dir.glob(
        f"produkt_klima_tag_*{station_code}*.txt"
    )
)

if len(recent_files) == 0:

    recent_files = list(
        recent_extract_dir.glob(
            "produkt_klima_tag_*.txt"
        )
    )

if len(recent_files) == 0:

    raise FileNotFoundError(
        "No recent produkt_klima_tag file found."
    )

recent_file = recent_files[0]

show_status("\nReading recent climate data:")
show_status(recent_file.name)


# =====================================================
# READ RECENT DATA AS df_dwd
# =====================================================

df_dwd = pd.read_csv(
    recent_file,
    sep=";",
    low_memory=False
)

df_dwd.columns = df_dwd.columns.str.strip()

df_dwd["MESS_DATUM"] = (
    df_dwd["MESS_DATUM"]
    .astype(str)
)

df_dwd = df_dwd.replace(
    [-999, -999.0, -9999, -9999.0],
    np.nan
)




# =====================================================
# REPLACE MISSING / BAD CLIMATE COLUMNS FROM NEARBY STATIONS
# =====================================================

required_columns = [
    "RSK",
    "VPM",
    "PM",
    "TMK",
    "UPM",
    "TXK",
    "TNK"
]

show_status("\nChecking required climate columns...")

# -----------------------------------------------------
# Clean main dataframe
# -----------------------------------------------------

df_dwd.columns = df_dwd.columns.str.strip()

df_dwd["MESS_DATUM"] = (
    df_dwd["MESS_DATUM"]
    .astype(str)
)

df_dwd = df_dwd.replace(
    [-999, -999.0, -9999, -9999.0],
    np.nan
)

# -----------------------------------------------------
# Function: check if a column has sufficient usable data
# -----------------------------------------------------

def column_is_good(df, col):

    if col not in df.columns:
        return False

    values = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    valid_count = values.notna().sum()
    nan_count = values.isna().sum()

    return (
        valid_count > 0
        and valid_count > nan_count
    )


# -----------------------------------------------------
# Identify bad / missing columns in original closest station
# -----------------------------------------------------

bad_columns = []

for col in required_columns:

    if not column_is_good(df_dwd, col):

        bad_columns.append(col)

        if col in df_dwd.columns:
            valid_count = pd.to_numeric(
                df_dwd[col],
                errors="coerce"
            ).notna().sum()

            nan_count = pd.to_numeric(
                df_dwd[col],
                errors="coerce"
            ).isna().sum()

            show_status(
                f"Column {col} is insufficient "
                f"({valid_count} valid, {nan_count} NaN)."
            )

        else:

            show_status(
                f"Column {col} is missing."
            )

if len(bad_columns) == 0:

    show_status(
        "All required climate columns are available "
        "in the closest DWD station."
    )

else:

    show_status(
        f"\nColumns to replace from nearby stations: "
        f"{bad_columns}"
    )


# =====================================================
# HELPER FUNCTION:
# DOWNLOAD + READ DATA FOR A STATION
# THEN DELETE DOWNLOADED / EXTRACTED FILES
# =====================================================

def load_klima_station_data(candidate_code):

    candidate_code = str(candidate_code).zfill(5)

    station_extract_dir = (
        download_dir / f"klima_replacement_{candidate_code}"
    )

    station_extract_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    hist_zip_file = None
    recent_zip_file = None

    try:

        # -------------------------------------------------
        # Historical file
        # -------------------------------------------------

        hist_page = requests.get(
            BASE_URL,
            timeout=30
        )

        hist_page.raise_for_status()

        hist_pattern = (
            rf"tageswerte_KL_{candidate_code}_.*?_hist\.zip"
        )

        hist_matches = re.findall(
            hist_pattern,
            hist_page.text
        )

        if len(hist_matches) == 0:

            raise FileNotFoundError(
                f"No historical KL file found for station "
                f"{candidate_code}"
            )

        hist_zip_filename = hist_matches[0]

        hist_zip_file = (
            download_dir / hist_zip_filename
        )

        urlretrieve(
            BASE_URL + hist_zip_filename,
            hist_zip_file
        )

        with zipfile.ZipFile(hist_zip_file, "r") as zip_ref:
            zip_ref.extractall(station_extract_dir)

        # -------------------------------------------------
        # Recent file
        # -------------------------------------------------

        recent_zip_filename = (
            f"tageswerte_KL_{candidate_code}_akt.zip"
        )

        recent_zip_file = (
            download_dir / recent_zip_filename
        )

        urlretrieve(
            RECENT_URL + recent_zip_filename,
            recent_zip_file
        )

        with zipfile.ZipFile(recent_zip_file, "r") as zip_ref:
            zip_ref.extractall(station_extract_dir)

        # -------------------------------------------------
        # Read all product files for this station
        # historical + recent
        # -------------------------------------------------

        product_files = list(
            station_extract_dir.glob(
                f"produkt_klima_tag_*{candidate_code}*.txt"
            )
        )

        if len(product_files) == 0:

            product_files = list(
                station_extract_dir.glob(
                    "produkt_klima_tag_*.txt"
                )
            )

        if len(product_files) == 0:

            raise FileNotFoundError(
                f"No produkt_klima_tag file found for station "
                f"{candidate_code}"
            )

        frames = []

        for file in product_files:

            df_tmp = pd.read_csv(
                file,
                sep=";",
                low_memory=False
            )

            df_tmp.columns = (
                df_tmp.columns.str.strip()
            )

            df_tmp["MESS_DATUM"] = (
                df_tmp["MESS_DATUM"]
                .astype(str)
            )

            df_tmp = df_tmp.replace(
                [-999, -999.0, -9999, -9999.0],
                np.nan
            )

            frames.append(df_tmp)

        df_candidate = (
            pd.concat(
                frames,
                ignore_index=True
            )
            .drop_duplicates(
                subset="MESS_DATUM",
                keep="last"
            )
            .sort_values("MESS_DATUM")
            .reset_index(drop=True)
        )

        return df_candidate

    finally:

        # -------------------------------------------------
        # Delete downloaded ZIP files
        # Heruntergeladene ZIP-Dateien löschen
        # -------------------------------------------------

        if hist_zip_file is not None and hist_zip_file.exists():

            try:
                hist_zip_file.unlink()
                show_status(f"  Deleted: {hist_zip_file.name}")

            except Exception as e:
                show_status(f"  Could not delete {hist_zip_file.name}: {e}")

        if recent_zip_file is not None and recent_zip_file.exists():

            try:
                recent_zip_file.unlink()
                show_status(f"  Deleted: {recent_zip_file.name}")

            except Exception as e:
                show_status(f"  Could not delete {recent_zip_file.name}: {e}")

        # -------------------------------------------------
        # Delete extracted station folder
        # Entpackten Stationsordner löschen
        # -------------------------------------------------

        if station_extract_dir.exists():

            try:
                shutil.rmtree(station_extract_dir)
                show_status(f"  Deleted folder: {station_extract_dir.name}")

            except Exception as e:
                show_status(f"  Could not delete folder {station_extract_dir.name}: {e}")


# =====================================================
# REPLACE BAD COLUMNS FROM NEAREST SUITABLE STATIONS
# =====================================================

if len(bad_columns) > 0:

    # Stations must already have distance calculated
    stations_sorted = (
        stations_df
        .sort_values("distance")
        .reset_index(drop=True)
    )

    replacement_info = {}

    for missing_col in bad_columns:

        show_status(
            f"\nSearching replacement source for column: "
            f"{missing_col}"
        )

        replacement_found = False

        for _, candidate_station in stations_sorted.iterrows():

            candidate_code = (
                candidate_station["Stations_id"]
            )

            candidate_name = (
                candidate_station["Stationsname"]
            )

            # Skip the original closest station
            if candidate_code == station_code:
                continue

            try:

                show_status(
                    f"Checking nearby station "
                    f"{candidate_name} ({candidate_code})..."
                )

                df_candidate = load_klima_station_data(
                    candidate_code
                )

                df_candidate.columns = (
                    df_candidate.columns.str.strip()
                )

                if missing_col not in df_candidate.columns:

                    show_status(
                        f"  {missing_col} not available. "
                        f"Skipping..."
                    )

                    continue

                df_candidate[missing_col] = pd.to_numeric(
                    df_candidate[missing_col],
                    errors="coerce"
                )

                # -----------------------------------------
                # Merge candidate column with df_dwd dates
                # -----------------------------------------

                df_replacement = df_candidate[
                    [
                        "MESS_DATUM",
                        missing_col
                    ]
                ].copy()

                df_replacement = (
                    df_replacement
                    .drop_duplicates(
                        subset="MESS_DATUM",
                        keep="last"
                    )
                )

                temp_col = (
                    f"{missing_col}_replacement"
                )

                df_replacement = df_replacement.rename(
                    columns={
                        missing_col: temp_col
                    }
                )

                df_test_merge = df_dwd[
                    ["MESS_DATUM"]
                ].merge(
                    df_replacement,
                    on="MESS_DATUM",
                    how="left"
                )

                valid_count = (
                    df_test_merge[temp_col]
                    .notna()
                    .sum()
                )

                nan_count = (
                    df_test_merge[temp_col]
                    .isna()
                    .sum()
                )

                if valid_count > 0 and valid_count > nan_count:

                    # -------------------------------------
                    # Replace whole bad column
                    # -------------------------------------

                    df_dwd = df_dwd.drop(
                        columns=[missing_col],
                        errors="ignore"
                    )

                    df_dwd = df_dwd.merge(
                        df_replacement,
                        on="MESS_DATUM",
                        how="left"
                    )

                    df_dwd = df_dwd.rename(
                        columns={
                            temp_col: missing_col
                        }
                    )

                    replacement_info[missing_col] = {
                        "station_code": candidate_code,
                        "station_name": candidate_name,
                        "valid_count": valid_count,
                        "nan_count": nan_count
                    }

                    show_status(
                        f"  Replacement accepted for {missing_col}: "
                        f"{candidate_name} ({candidate_code}) "
                        f"with {valid_count} valid and "
                        f"{nan_count} NaN values."
                    )

                    replacement_found = True

                    break

                else:

                    show_status(
                        f"  {missing_col} insufficient at "
                        f"{candidate_name} ({candidate_code}): "
                        f"{valid_count} valid, {nan_count} NaN."
                    )

            except Exception as e:

                show_status(
                    f"  Station {candidate_code} failed: {e}"
                )

                continue

        if not replacement_found:

            show_status(
                f"\nWARNING: No suitable replacement found "
                f"for column {missing_col}."
            )


# =====================================================
# FINAL CHECK
# =====================================================

show_status("\nFinal required-column status:")

for col in required_columns:

    if col in df_dwd.columns:

        values = pd.to_numeric(
            df_dwd[col],
            errors="coerce"
        )

        valid_count = values.notna().sum()
        nan_count = values.isna().sum()

        show_status(
            f"{col}: {valid_count} valid, "
            f"{nan_count} NaN"
        )

    else:

        show_status(
            f"{col}: missing"
        )

show_status("\nReplacement summary:")

if "replacement_info" in locals() and len(replacement_info) > 0:

    for col, info in replacement_info.items():

        show_status(
            f"{col} replaced from "
            f"{info['station_name']} "
            f"({info['station_code']})"
        )

else:

    show_status("No replacements were necessary.")




# =====================================================
# FIND CLOSEST WIND STATION if no daily mean wind data is available in df_dwd
# =====================================================

if (
    "FM" not in df_dwd.columns
    or df_dwd["FM"].notna().sum() < df_dwd["FM"].isna().sum()
):

    df_dwd = df_dwd.drop(
        columns=["FM", "FX", "QN_3", "QN_4"],
        errors="ignore"
    )

    WIND_RECENT_URL = (
        "https://opendata.dwd.de/climate_environment/CDC/"
        "observations_germany/climate/hourly/wind/recent/"
    )

    WIND_STATION_FILE = "FF_Stundenwerte_Beschreibung_Stationen.txt"

    show_status("\nLoading recent wind station information...")

    wind_station_local = download_dir / WIND_STATION_FILE

    urlretrieve(
        WIND_RECENT_URL + WIND_STATION_FILE,
        wind_station_local
    )

    wind_stations = []

    with open(wind_station_local, "r", encoding="latin1") as f:

        for line in f:

            if line.startswith("Stations_id"):
                continue

            if line.startswith("-"):
                continue

            if not line.strip():
                continue

            parts = line.strip().split()

            if len(parts) < 8:
                continue

            try:

                station_id = parts[0].zfill(5)

                geo_breite = float(parts[4])
                geo_laenge = float(parts[5])

                station_name = " ".join(parts[6:-1])

                wind_stations.append(
                    {
                        "Stations_id": station_id,
                        "Stationsname": station_name,
                        "geoBreite": geo_breite,
                        "geoLaenge": geo_laenge,
                    }
                )

            except Exception:
                continue

    wind_df = pd.DataFrame(wind_stations)

    # Optional: delete downloaded station-description file after reading
    try:
        wind_station_local.unlink()
        show_status(f"Deleted: {wind_station_local.name}")
    except Exception as e:
        show_status(f"Could not delete {wind_station_local.name}: {e}")

    # =====================================================
    # FIND STATIONS WITH RECENT WIND ZIP FILES
    # =====================================================

    show_status("\nChecking available recent wind stations...")

    recent_page = requests.get(
        WIND_RECENT_URL,
        timeout=30
    )

    recent_page.raise_for_status()

    recent_codes = set(
        re.findall(
            r"stundenwerte_FF_(\d{5})_akt\.zip",
            recent_page.text
        )
    )

    wind_df = wind_df[
        wind_df["Stations_id"].isin(recent_codes)
    ].copy()

    if len(wind_df) == 0:

        raise RuntimeError(
            "No wind stations with recent data were found."
        )

    # =====================================================
    # FIND CLOSEST WIND STATION
    # =====================================================

    wind_df["distance"] = np.sqrt(
        (wind_df["geoBreite"] - lat) ** 2
        + (wind_df["geoLaenge"] - lon) ** 2
    )

    wind_df = (
        wind_df
        .sort_values("distance")
        .reset_index(drop=True)
    )

    closest_wind_station = wind_df.iloc[0]

    wind_station_code = (
        closest_wind_station["Stations_id"]
    )

    wind_station_name = (
        closest_wind_station["Stationsname"]
    )

    show_status(
        f"\nClosest recent wind station: "
        f"{wind_station_name} "
        f"({wind_station_code})"
    )

    # =====================================================
    # DOWNLOAD RECENT WIND FILE ONLY
    # =====================================================

    wind_recent_filename = (
        f"stundenwerte_FF_{wind_station_code}_akt.zip"
    )

    wind_recent_file = (
        download_dir / wind_recent_filename
    )

    wind_extract_dir = (
        download_dir / f"wind_recent_{wind_station_code}"
    )

    wind_extract_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        show_status(
            f"\nDownloading recent wind data:"
            f"\n{wind_recent_filename}"
        )

        urlretrieve(
            WIND_RECENT_URL + wind_recent_filename,
            wind_recent_file
        )

        # =====================================================
        # EXTRACT RECENT WIND ZIP
        # =====================================================

        show_status("\nExtracting recent wind ZIP file...")

        with zipfile.ZipFile(wind_recent_file, "r") as z:
            z.extractall(wind_extract_dir)

        show_status("Recent wind ZIP extraction completed.")

        # =====================================================
        # FIND RECENT WIND DATA FILE
        # =====================================================

        wind_txt_candidates = list(
            wind_extract_dir.glob(
                f"produkt_ff_stunde_*_{wind_station_code}.txt"
            )
        )

        if len(wind_txt_candidates) == 0:

            wind_txt_candidates = list(
                wind_extract_dir.glob(
                    "produkt_ff_stunde_*.txt"
                )
            )

        if len(wind_txt_candidates) == 0:

            raise FileNotFoundError(
                "No recent wind product file found."
            )

        wind_recent_txt = wind_txt_candidates[0]

        show_status(
            f"\nReading recent wind file:"
            f"\n{wind_recent_txt.name}"
        )

        df_wind_recent = pd.read_csv(
            wind_recent_txt,
            sep=";",
            low_memory=False
        )

    finally:

        # =====================================================
        # DELETE DOWNLOADED / EXTRACTED WIND FILES
        # =====================================================

        if wind_recent_file.exists():

            try:
                wind_recent_file.unlink()
                show_status(f"Deleted: {wind_recent_file.name}")

            except Exception as e:
                show_status(
                    f"Could not delete "
                    f"{wind_recent_file.name}: {e}"
                )

        if wind_extract_dir.exists():

            try:
                shutil.rmtree(wind_extract_dir)
                show_status(f"Deleted folder: {wind_extract_dir.name}")

            except Exception as e:
                show_status(
                    f"Could not delete folder "
                    f"{wind_extract_dir.name}: {e}"
                )

    # =====================================================
    # PREPARE RECENT WIND DATA
    # =====================================================

    df_wind_recent.columns = (
        df_wind_recent.columns.str.strip()
    )

    df_wind_recent.replace(
        [-999, -9999, -999.0, -9999.0],
        np.nan,
        inplace=True
    )

    df_wind_recent["MESS_DATUM"] = pd.to_datetime(
        df_wind_recent["MESS_DATUM"].astype(str),
        format="%Y%m%d%H",
        errors="coerce"
    )

    # =====================================================
    # DETERMINE WIND SPEED COLUMN
    # =====================================================

    if "F" in df_wind_recent.columns:

        wind_speed_col = "F"

    elif "FF" in df_wind_recent.columns:

        wind_speed_col = "FF"

    else:

        raise KeyError(
            "No wind-speed column found. "
            "Expected column 'F' or 'FF'."
        )

    df_wind_recent[wind_speed_col] = pd.to_numeric(
        df_wind_recent[wind_speed_col],
        errors="coerce"
    )

    # =====================================================
    # CALCULATE DAILY MEAN WIND SPEED
    # =====================================================

    df_wind_daily = (
        df_wind_recent
        .dropna(subset=["MESS_DATUM"])
        .set_index("MESS_DATUM")
        .resample("D")
        .agg({wind_speed_col: "mean"})
        .reset_index()
    )

    df_wind_daily.rename(
        columns={wind_speed_col: "FM"},
        inplace=True
    )

    df_wind_daily["MESS_DATUM"] = (
        df_wind_daily["MESS_DATUM"]
        .dt.strftime("%Y%m%d")
    )

    show_status(
        f"Daily recent wind dataset contains "
        f"{len(df_wind_daily):,} days."
    )

    # =====================================================
    # MERGE WIND INTO DWD DATA
    # =====================================================

    df_dwd["MESS_DATUM"] = (
        df_dwd["MESS_DATUM"]
        .astype(str)
    )

    df_dwd = df_dwd.merge(
        df_wind_daily[
            ["MESS_DATUM", "FM"]
        ],
        on="MESS_DATUM",
        how="left"
    )

    show_status(
        "\nRecent wind data successfully merged."
    )


else:

    show_status(
        "\nDaily mean wind data FM is already available "
        "and sufficiently complete in main station."
    )






gamma=0.067 #-KPa/degC psycometric constant for elevation 0-100m (FAO Annex2 table 2.2) 

r=0.08  #- Albedo for water
z0=0.03 #- cm roughness of water surface for  (0.01-0.06) from Chow's hydrology Table 2.8.2
z2=2 #- height of windspeed measurement

#-Priestley-Taylor method (Chow's book eq. 3.5.27)
alpha_evap=0.8


fmt = '%Y-%m-%d' # date format
fmt_de = '%d.%m.%Y' # date format German
#-- crop coefficients

rho_w=997 #-density of water

Kc_ini_grass=0.35   #-- initial phase (Bermuda grass)
Kc_mid_grass=0.9    #-- middle phase
Kc_end_grass=0.65   #-- final phase
Dini_grass=61
Lini_grass=10
Ldev_grass=25       #-development phase
Lmid_grass=35
Lend_grass=105

Kc_ini_reed=1.0     #-- Reed swamp standing water
Kc_mid_reed=1.2
Kc_end_reed=1.0
Dini_reed=122
Lini_reed=10
Ldev_reed=30
Lmid_reed=80
Lend_reed=20

Kc_ini_tree=0.5     #-- wallnut tree
Kc_mid_tree=1.0
Kc_end_tree=0.6518
Dini_tree=92
Lini_tree=20
Ldev_tree=10
Lmid_tree=130
Lend_tree=30

################################################################################
pi=math.pi


#- fill missing data with NaN and then interpolate the missing values
df_dwd['date']=pd.to_datetime(df_dwd['MESS_DATUM'])

##########################################################################
#- compute open water evaporation
##########################################################################

G=0 #-soil heat flux, zero (ignored) for daily calculations

Lz=360-(int(lon/15.0)*15)      #- longitude of the center of local time zone [degrees west of Greenwich]
Lm = 360-lon     #- longitude of the measurement site [degrees west of Greenwich]
kRs = 0.16 #-0.16 for inland locations, 0.19 for coastal locations


#-calculate vapour pressure deficit (es - ea) (chapter 3, box 7)
df_dwd['eTmax']=0.6108*np.exp((17.27*df_dwd['TXK'])/(df_dwd['TXK']+237.3)) #-kPa
df_dwd['eTmin']=0.6108*np.exp((17.27*df_dwd['TNK'])/(df_dwd['TNK']+237.3)) #-kPa
#df_dwd['es']=(df_dwd['eTmax']+df_dwd['eTmin'])/2 #-saturation vapour pressure kPa (FAO method)
df_dwd['es']=0.1*df_dwd['VPM']/(df_dwd['UPM']/100) #-saturation vapour pressure kPa 
df_dwd['vap_deficit']=df_dwd['es']-df_dwd['VPM']*0.1 #-calculate vapour pressure deficit (es - ea) kPa
df_dwd['delta']=(4098*(0.6108*np.exp((17.27*df_dwd['TMK'])/df_dwd['TMK']+273.3)))\
    /((df_dwd['TMK']+237.3)**2) #-Slope of saturation vapour pressure curve (Delta capital)

#- calculate solar radiation
df_dwd.date=pd.to_datetime(df_dwd.date)
df_dwd['doy']=df_dwd['date'].dt.dayofyear # day of year
df_dwd['dr']=1+0.033*np.cos(2*pi*df_dwd['doy']/365) # inverse relative distance Earth-Sun
df_dwd['dlt']=0.409*np.sin(2*pi*df_dwd['doy']/365-1.39) # solar declination, delta (small)
phi = lat*pi/180 #- lat in radians
df_dwd['omega_s']=np.arccos(-np.tan(phi)*np.tan(df_dwd['dlt'])) # sunset hour angle
df_dwd['Ra']=(24*60)*0.0820*df_dwd['dr']*(df_dwd['omega_s']*np.sin(phi)*np.sin(df_dwd['dlt'])+
    np.cos(phi)*np.cos(df_dwd['dlt'])*np.sin(df_dwd['omega_s']))/math.pi # extraterrestrial radiation
df_dwd['N'] = (24 / math.pi) * df_dwd['omega_s'] # Maximum possible sunshine duration N [hours]
df_dwd['Rso']=(0.75+2e-5*elev_st)*df_dwd['Ra'] #clear-sky radiation [MJ m-2 day-1]
temp_range = (df_dwd['TXK'] - df_dwd['TNK']).clip(lower=0)

df_dwd['Rs'] = (kRs * np.sqrt(temp_range)* df_dwd['Ra']) # shortwave radiation
df_dwd['Rs'] = np.minimum(df_dwd['Rs'],df_dwd['Rso']) # Rs should not be larger than clear-sky rad Rso
df_dwd['Rns']=(1-0.23)*df_dwd['Rs'] # Net solar or net shortwave radiation 
# 0.23=alpha albedo or canopy reflection coefficient,grass reference crop

df_dwd['Rnl']=4.903e-9*(((df_dwd['TXK']+273.16)+(df_dwd['TNK']+273.16))/2)*\
(0.34-0.14*np.sqrt(df_dwd['VPM']*0.1))*((1.35*df_dwd['Rs']/df_dwd['Rso'])-0.35) #- net longwave radiation
df_dwd['Rn']=df_dwd['Rns']-df_dwd['Rnl'] #- Net radiation

#-compute ETc (Penman-Montith eq.)
df_dwd['ET0']=\
    (0.408*df_dwd['delta']*(df_dwd['Rn']-G)+gamma*(900/(df_dwd['TMK']+273))*df_dwd['FM']*df_dwd['vap_deficit'])\
    /(df_dwd['delta']+gamma*(1+0.34*df_dwd['FM'])) #- Penman-Monteith equation, ET for reference crop

#- compute Eo 
#-Penman equation from Chow's book eqs. 3.5.10-3.5.26
#- latent heat (lambda)=0.40
k=0.4 #- von Karman constant (Chow's book, eq. 3.5.14) 
df_dwd['rho_a']=(df_dwd['PM']*100*0.0289652+df_dwd['VPM']*100*0.018016)/\
    (8.31446*(df_dwd['TMK']+273)) #- kg/m3 density of humid air as function of pressure (Pa) and temp.
df_dwd['B']=(0.622*(k**2)*df_dwd['rho_a']*df_dwd['FM'])/(df_dwd['PM']*100*rho_w*(np.log(z2*100/z0))**2) #- Chow's book, eq. 3.5.18
df_dwd['Ea']=df_dwd['B']*(df_dwd['es']*1000-df_dwd['VPM']*100) #- Chow's book, eq. 3.5.17
df_dwd['Er']=0.0353*df_dwd['Rn']

df_dwd['Eo_penman']= (df_dwd['delta']/(df_dwd['delta']+gamma))*df_dwd['Er']+\
    (gamma/(df_dwd['delta']+gamma))*df_dwd['Ea']#- mm/d Penman Eq. from Chow's book eq. 3.5.26
df_dwd.Eo_penman=df_dwd.Eo_penman.mask(df_dwd.Eo_penman.lt(0),0) #- set very small negative evaporations to 0

#-Priestley-Taylor method (Chow's book eq. 3.5.27)
df_dwd['Eo_prst']=alpha_evap*(df_dwd['delta']/(df_dwd['delta']+gamma))*df_dwd['Er']
df_dwd.Eo_prst=df_dwd.Eo_prst.mask(df_dwd.Eo_prst.lt(0),0) #- set very small negative evaporations to 0


#with tqdm(total=df_out.shape[0], ascii=' █') as pbar:
#    for i, row in df_out.iloc[1:nrows].iterrows():  # Iter begins from second row of dataframe (first row as initial conditions)
#        pbar.update(1)
#        pbar.set_description('%s' %row['date'])
 


# CREATE RESULTS DIRECTORY
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


df_dwd['water_bal_penman']=df_dwd['RSK']-df_dwd['Eo_penman']
df_dwd['water_bal_prst']=df_dwd['RSK']-df_dwd['Eo_prst']
df_dwd.to_excel('results/res_All_DWD_station_%s_%s.xlsx'%(station_code,station_name_final), index=False)

df_dwd=df_dwd[['date','RSK','Eo_penman','water_bal_penman','Eo_prst','water_bal_prst']]
df_dwd.to_excel('results/res_Evap_DWD_station_%s_%s.xlsx'%(station_code,station_name_final), index=False)

#monthly means:
df_dwd_monthly = (df_dwd.set_index("date").resample("ME").mean(numeric_only=True).reset_index())
df_dwd_monthly.to_excel('results/res_Evap_monthly_DWD_station_%s_%s.xlsx'%(station_code,station_name_final), index=False)


print(
        f"\nClosest DWD climate station with recent data:"
        f"\nStation name : {station_name_final}"
        f"\nStation code : {station_code_final}"
        f"\nLast data    : {closest_station['bis_datum'].date()}"
    )

print('Evaporation Calculation completed. See ')
