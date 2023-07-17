import numpy as np

def calculate_mean_std(value_list):
    mean = np.mean(value_list)
    std = np.std(value_list, ddof=1)
    return "$" + f"{mean:.3f}_{{\pm{std:.3f}}}" + "$"

value_list = [0.43278, 0.4333, 0.424]
result = calculate_mean_std(value_list)

print(result)
