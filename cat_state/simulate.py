import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed

from cat_state.cat_state_generation import cat_state_FT
from cat_state.circuit_extraction import make_stim_circ_noisy


def init_data_folder():
    Path("simulation_data").mkdir(parents=True, exist_ok=True)


def save_simulation_data(n, t, samples: np.ndarray):
    # now = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = f"simulation_data/samples_{n}_{t}.npz"
    np.savez_compressed(file_name, samples)
    print(f"Saved samples to {file_name}")


def run_simulation(n: int, t: int, p: float, num_samples: int = 1_000_000, save_samples: bool = False):
    init_data_folder()

    circ = cat_state_FT(n, t, run_verification=False)
    if circ is None:
        return

    num_flags = circ.num_qubits - n
    noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=0, p_meas=2 / 3 * p)
    noisy_circ.append("M", range(num_flags, circ.num_qubits))

    circuit_sampler = noisy_circ.compile_sampler()
    samples: np.ndarray = circuit_sampler.sample(num_samples)

    error_detected = np.any(samples[:, :num_flags], axis=1)
    post_selected_samples = samples[~error_detected]
    num_data_errors = np.sum(post_selected_samples[:, num_flags:], axis=1)
    num_data_errors = np.where(num_data_errors > n // 2, n - num_data_errors, num_data_errors)
    unique, counts = np.unique(num_data_errors, return_counts=True)
    errors_summary = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Number of data errors for {t}-fault-tolerant {n}-cat-state: {errors_summary}")

    if save_samples:
        save_simulation_data(n, t, samples)

    return errors_summary


def visualise(collected_data, t):
    df = pd.DataFrame(collected_data)

    # ---------------------------------------------------------
    # 2. Filtering & Preparation
    # ---------------------------------------------------------

    # Filter for only 1 <= k <= 5
    df_filtered = df[df['error_count'].between(1, 5)].copy()

    # Ensure 'error_count' is treated as a category for the legend/colors, not a continuous number
    df_filtered['k_label'] = df_filtered['error_count'].apply(lambda x: f"k={x}")

    # ---------------------------------------------------------
    # 3. Visualization (Matching the Screenshot)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6), dpi=120)

    # Create the plot using the 'viridis' palette (Dark Blue -> Green -> Yellow)
    sns.lineplot(
        data=df_filtered,
        x='n',
        y='probability',
        hue='k_label',
        style='k_label',  # Ensures markers are drawn
        markers=['o'] * 5,  # Force circle markers for all
        dashes=False,  # Solid lines
        palette='viridis',  # The specific color map you asked for
        markersize=8,
        linewidth=2
    )

    # Y-Axis Log Scale
    plt.yscale('log')

    # Axis Labels
    plt.xlabel("Cat State Size", fontsize=12)
    plt.ylabel("$P_k$", fontsize=12)

    # Grid Styling (Light dashed lines)
    plt.grid(True, which="both", ls="-", color='lightgrey', alpha=0.5)

    # Legend Styling (Top, Horizontal, No Box)
    # bbox_to_anchor moves it above the plot, ncol=5 makes it horizontal
    plt.legend(
        title="",
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(f"simulation_data/cat_state_{t}.png", dpi=1200)
    plt.show()


# 1. Define a helper function that handles a SINGLE 'n'
def process_simulation(n, t, p, num_samples):
    # Run your simulation
    data = run_simulation(n, t, p, num_samples)

    if data is None:
        return []

    total_counts = sum(data.values())
    results = []

    # Process the data locally for this thread/process
    for errors, count in data.items():
        results.append({
            'n': n,
            'error_count': errors,
            'probability': count / total_counts
        })

    return results


if __name__ == "__main__":
    start_time = time.time()

    collected_data = []

    print("Starting simulation loop...")

    parallel_results = Parallel(n_jobs=-2)(
        delayed(process_simulation)(n, t=3, p=0.01, num_samples=100_000_000) for n in range(8, 36)
    )
    collected_data = [item for sublist in parallel_results for item in sublist]

    print("Simulation complete")
    visualise(collected_data, 3)

    print("--- %s seconds ---" % (time.time() - start_time))
