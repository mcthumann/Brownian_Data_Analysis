import matplotlib
import matplotlib.ticker as ticker
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
def plot_results(trapped_microsphere_results, label, psd=True, pacf=False):
    plt.gca().yaxis.set_major_formatter(ticker.LogFormatter(base=10))
    if psd:
        plot_psd(trapped_microsphere_results, label="PSD " + label)
    if pacf:
        plot_pacf(trapped_microsphere_results, label="PACF " + label)

def plot_psd(dataset, label, avg=True):
    all_responses = np.array([item["responses"][1:-1] for item in dataset])
    print("averaged " + str(len(dataset)))
    avg_response = np.mean(all_responses, axis=0)
    plt.plot(dataset[0]["frequency"][1:-1], avg_response, label=label,
             linewidth=.25)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("Trapped Microsphere Signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Signal [V^2/Hz]")

def plot_pacf(dataset, label):
    for result in dataset:
        plt.plot(result["time"], result["acf"], label=label,
                 linewidth=1.5)
        plt.xscale("log")
        plt.legend()
