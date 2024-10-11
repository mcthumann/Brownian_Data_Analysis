import matplotlib.pyplot as plt
def plot_results(bright_noise_results, trapped_microsphere_results, plot_together=True):
    if plot_together:
        # Plot both datasets on the same graph
        plt.figure(figsize=(10, 6))
        #plot_dataset(bright_noise_results, label="Bright Noise", linestyle='--', linewidth=1)
        plot_dataset(trapped_microsphere_results, label="Trapped Microsphere", linestyle='--', linewidth=1)
        plt.legend()
        plt.title("Combined Plot of Bright Noise and Trapped Microsphere Signal")
    else:
        # Plot each dataset on separate graphs
        plt.figure(figsize=(10, 6))
        #plot_dataset(bright_noise_results, label="Bright Noise", linestyle='--', linewidth=1)
        plt.legend()
        plt.title("Bright Noise")

        plt.figure(figsize=(10, 6))
        plot_dataset(trapped_microsphere_results, label="Trapped Microsphere", linestyle='-', linewidth=1)
        plt.legend()
        plt.title("Trapped Microsphere Signal")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Signal [V^2/Hz]")

def plot_dataset(dataset, label, linestyle, linewidth):
    for result in dataset:
        plt.plot(result["frequency"][1:-1], result["responses"][1:-1], label=label, linestyle=linestyle,
                 linewidth=linewidth)
