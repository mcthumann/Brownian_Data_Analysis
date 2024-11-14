from nptdms import TdmsFile
import os
import pandas as pd
import pickle
from config import SIM

def read_tdms_file(file_path, data_col):
    tdms_file = TdmsFile.read(file_path)
    sample_rate = tdms_file["main"].properties.get("R", None)
    series = tdms_file["main"]["X_" + str(data_col-1)]
    track_len = len(series.data)
    config_args = {
        "sampling_rate": sample_rate,
        "track_len": track_len
    }
    return series[:], config_args

def read_csv_file(file_path, data_col):
    # Read CSV with low_memory=False to avoid DtypeWarning
    df = pd.read_csv(file_path, low_memory=False)

    # Filter columns that start with "Position"
    position_columns = [col for col in df.columns if col.startswith("Position")]

    if data_col - 1 < len(position_columns):
        position_col = position_columns[data_col - 1]
        series = df[position_col].iloc[:]
        # Extract arguments from CSV
        a = df["a"].iloc[0] if "a" in df.columns else 5e-6
        eta = df["eta"].iloc[0] if "eta" in df.columns else 1e-3
        rho_silica = df["rho_silica"].iloc[0] if "rho_silica" in df.columns else 2200
        rho_f = df["rho_f"].iloc[0] if "rho_f" in df.columns else 1000
        sampling_rate = df["sampling_rate"].iloc[0] if "sampling_rate" in df.columns else 10000
        track_len = len(series)  # Length of the series
        stop = df["stop"].iloc[0] if "stop" in df.columns else None
        start = df["start"].iloc[0] if "start" in df.columns else None

        config_args = {
            "a": a,
            "eta": eta,
            "rho_silica": rho_silica,
            "rho_f": rho_f,
            "sampling_rate": sampling_rate,
            "track_len": track_len,
            "stop": stop,
            "start": start,
        }

        return series, config_args
    else:
        raise ValueError(f"Data column index {data_col} is out of range for available 'Position' columns.")

def process_folder(folder_name, tracks_per_file, num_traces):
    results = []
    for i in range(num_traces):
        print("Reading ", folder_name, str(i))
        result = process_file(folder_name, i, tracks_per_file)
        if result:
            results.append(result)
    return results

def process_file(folder_name, trace_num, data_col):
    if SIM:
        series, args = read_csv_file(folder_name, trace_num)
    else:
        file_path = os.path.join(folder_name, "iter_" + str(trace_num) + ".tdms")
        series, args = read_tdms_file(file_path, data_col)

    return {
        "series": series,
        "args": args
    }

def save_results(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def check_and_load_or_process(filename, *args):
    if os.path.exists(filename):
        print(f"Loading results from {filename}")
        return load_results(filename)
    else:
        print(f"Processing data for {filename}")
        results = process_folder(*args)
        # Returns a list of args, and traces
        return results