import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import platform
from config import ACF

if platform.system() == "Darwin":  # mac is identified as 'Darwin'
    matplotlib.use('TkAgg')

def plot_results(results, label, psd=True):
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatter(base=10))
    if psd:
        # plot_best_psds(results, "PSD " + str(10), 10)
        # plot_best_psds(results, "PSD " + str(20), 20)
        # plot_best_psds(results, "PSD " + str(30), 30)
        # plot_best_psds(results, "PSD " + str(50), 50)
        plot_psd(results, label="PSD " + label)
        plt.show()
    if ACF:
        plot_pacf(results, label="PACF " + label)
        plt.show()

        plot_vacf(results, label="VACF " + label)
        plt.show()

def plot_psd(dataset, label, avg=True):
    if avg:
        all_responses = np.array([item["psd"][1:-1] for item in dataset])
    else:
        all_responses = dataset[0]["psd"][1:-1]
    print("averaged " + str(len(dataset)) + " PSDs")
    avg_response = np.mean(all_responses, axis=0)
    plt.plot(dataset[0]["frequency"][1:-1], avg_response, label=label,
             linewidth=.25)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("Power Spectral Density Data")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Signal [V^2/Hz]")

def plot_best_psds(dataset, label, top_percentage):

    sums = [np.sum(data["psd"][1:1000]) for data in dataset]

    num_top_arrays = int(len(dataset) * (top_percentage / 100))

    # Get the indices of arrays sorted by their sums (in descending order)
    top_indices = np.argsort(sums)[-num_top_arrays:]

    # Select the top percentage arrays
    top_psds = [dataset[i]["psd"][1:-1] for i in top_indices]

    print("Averaged " + str(num_top_arrays) + " PSDs")

    # Calculate the average response across the top PSDs
    avg_response = np.mean(top_psds, axis=0)

    plt.plot(dataset[0]["frequency"][1:-1], avg_response, label=label,
             linewidth=.25)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("Best " + str(top_percentage) + " Percent of Data Power Spectral Density")
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

def plot_vacf(dataset, label, avg=True):
    if avg:
        all_vacf = np.array([item["v_acf"][:] for item in dataset])
        vacf = np.mean(all_vacf, axis=0)
        print("averaged " + str(len(dataset)) + " VACF Power Spectral Density")
    else:
        vacf = np.array(dataset[0]["v_acf"][:])

    plt.plot(vacf, label=label,
             linewidth=1)
    plt.xscale("log")
    plt.legend()
    plt.title("VACF")
    plt.xlabel("Time")
    plt.ylabel("VACF")