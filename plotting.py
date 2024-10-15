import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import platform

if platform.system() == "Darwin":  # mac is identified as 'Darwin'
    matplotlib.use('TkAgg')

def plot_results(results, label, psd=True, pacf=False):
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatter(base=10))
    if psd:
        plot_psd(results, label="PSD " + label)
    if pacf:
        plot_pacf(results, label="PACF " + label)

def plot_psd(dataset, label, avg=True):
    if avg:
        all_responses = np.array([item["responses"][1:-1] for item in dataset])
    else:
        all_responses = dataset[0]["responses"][1:-1]
    print("averaged " + str(len(dataset)) + " PSDs")
    avg_response = np.mean(all_responses, axis=0)
    plt.plot(dataset[0]["frequency"][1:-1], avg_response, label=label,
             linewidth=.25)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Signal [V^2/Hz]")

def plot_pacf(dataset, label, avg=True):
    if avg:
        all_acf = np.array([item["acf"][:] for item in dataset])
    else:
        all_acf = dataset[0]["acf"][:]
    avg_acf = np.mean(all_acf, axis=0)
    plt.plot(dataset[0]["time"][:], avg_acf, label=label,
             linewidth=1)
    plt.xscale("log")
    plt.legend()
    plt.title("Position Autocorrelation Function")
    plt.xlabel("Normalized PACF")
    plt.ylabel("Time")