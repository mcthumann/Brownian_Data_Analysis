# Read in text file
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load data from files
columns = ["High Pass", "R Fit", "V Fit", "K Fit"]

# Read the two files
df_lower = pd.read_csv("../data/tikonovsearch_lower.dat", delim_whitespace=True, comment='#', names=columns)
df_upper = pd.read_csv("../data/tikonovsearch.dat", delim_whitespace=True, comment='#', names=columns)

# Combine and sort
df_combined = pd.concat([df_lower, df_upper], ignore_index=True)
df_combined.sort_values(by="High Pass", inplace=True)

# Create NumPy arrays
high_pass = df_combined["High Pass"].values
r_fit = df_combined["R Fit"].values
v_fit = df_combined["V Fit"].values
k_fit = df_combined["K Fit"].values

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(high_pass, r_fit, label="R Fit")
plt.show()
plt.plot(high_pass, v_fit, label="V Fit")
plt.show()
plt.plot(high_pass, k_fit, label="K Fit")

plt.show()


print(np.max(r_fit)-r_fit[15])
print(np.max(v_fit)-v_fit[15])
print(k_fit[15] - np.min(k_fit))
