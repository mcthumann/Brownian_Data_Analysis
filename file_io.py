from nptdms import TdmsFile
import os
import pandas as pd
import pickle
from config import SIM, SAMPLE

def read_tdms_file(file_path, data_col, trace_idx):
    tdms_file = TdmsFile.read(file_path)
    sample_rate = tdms_file["main"].properties.get("r", None)
    series = tdms_file["main"][data_col + "_" + str(trace_idx)]
    track_len = len(series.data)
    config_args = {
        "sampling_rate": sample_rate,
        "track_len": track_len
    }

    return series[:], config_args

def read_csv_file(file_path, trace_num):
    # Read CSV with low_memory=False to avoid DtypeWarning
    df = pd.read_csv(file_path, low_memory=False)

    # Filter columns that start with "Position"
    position_columns = [col for col in df.columns if col.startswith("Position")]

    if trace_num - 1 < len(position_columns):
        position_col = position_columns[trace_num - 1]
        series = df[position_col].iloc[:]
        series = series[::SAMPLE]
        # Extract arguments from CSV
        d = df["d"].iloc[0] if "d" in df.columns else 2e-6
        eta = df["eta"].iloc[0] if "eta" in df.columns else 1e-3
        rho_silica = df["rho_silica"].iloc[0] if "rho_silica" in df.columns else 2200
        rho_f = df["rho_f"].iloc[0] if "rho_f" in df.columns else 1000
        sampling_rate = df["sampling_rate"].iloc[0] if "sampling_rate" in df.columns else 10000
        sampling_rate /= SAMPLE
        track_len = len(series)  # Length of the series
        # stop = df["stop"].iloc[0] if "stop" in df.columns else None
        # start = df["start"].iloc[0] if "start" in df.columns else None

        config_args = {
            "d": d,
            "eta": eta,
            "rho_silica": rho_silica,
            "rho_f": rho_f,
            "sampling_rate": sampling_rate,
            "track_len": track_len,
            # "stop": stop,
            # "start": start,
        }

        return series, config_args
    else:
        raise ValueError(f"Data column index {trace_num} is out of range for available 'Position' columns.")

def process_folder(offset, folder_name, data_col, num_traces, traces_per):
    results = []
    for i in range(num_traces):
        print("Reading ", folder_name, str(i))
        print("data_col ", data_col)
        for j in range(traces_per):
            result = process_file(folder_name, i, data_col, j, offset=offset)
            if result:
                results.append(result)
    return results

def process_file(folder_name, trace_num, data_col, trace_idx, offset):
    if SIM:
        series, args = read_csv_file(folder_name, trace_num)
    else:
        trace_num = trace_num + offset
        file_path = os.path.join(folder_name, "iter_" + str(trace_num) + ".tdms")
        series, args = read_tdms_file(file_path, data_col, trace_idx)

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

def check_and_load_or_process(filename, offset, *args,):
    if os.path.exists(filename):
        print(f"Loading results from {filename}")
        return load_results(filename)
    else:
        print(f"Processing data for {filename}")
        results = process_folder(offset, *args)
        # Returns a list of args, and traces
        return results