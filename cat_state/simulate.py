import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import norm

from cat_state.cat_state_generation import cat_state_FT
from cat_state.circuit_extraction import make_stim_circ_noisy


def init_data_folder():
    Path("simulation_data").mkdir(parents=True, exist_ok=True)


def save_simulation_data(n, t, samples: np.ndarray):
    # now = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = f"simulation_data/samples_{n}_{t}.npz"
    np.savez_compressed(file_name, samples)
    print(f"Saved samples to {file_name}")


def calculate_wilson_interval(k, n, confidence=0.95):
    """
    Calculates the Wilson Score Interval for a binomial proportion.
    Ideal for QEC where error rates (p) are often close to 0.
    k: number of successes (or errors)
    n: total number of trials (accepted samples)
    """
    if n == 0:
        return 0.0, 0.0

    p = k / n
    z = norm.ppf(1 - (1 - confidence) / 2).tolist()

    denominator = 1 + z ** 2 / n
    center_adjusted_prob = p + z ** 2 / (2 * n)
    adjusted_std_dev = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)

    lower_bound = (center_adjusted_prob - adjusted_std_dev) / denominator
    upper_bound = (center_adjusted_prob + adjusted_std_dev) / denominator

    return max(0.0, lower_bound), min(1.0, upper_bound)


def process_samples(total_samples_attempted: int, samples: np.ndarray, num_flags: int, n: int, t: int):
    """
    Processes raw simulation samples to generate rich statistics.
    """
    # 1. Post-selection (Discard runs where flags triggered)
    if samples.shape[0] > 0:
        error_detected = np.any(samples[:, :num_flags], axis=1)
        post_selected_samples = samples[~error_detected]
    else:
        post_selected_samples = np.array([])

    num_accepted = post_selected_samples.shape[0]
    acceptance_rate = num_accepted / total_samples_attempted if total_samples_attempted > 0 else 0.0

    # 2. Calculate Data Errors (Distance from nearest codeword)
    if num_accepted > 0:
        raw_errors = np.sum(post_selected_samples[:, num_flags:], axis=1)
        # Fold errors: if > n/2, it means we drifted to the "other" logical state
        num_data_errors = np.where(raw_errors > n // 2, n - raw_errors, raw_errors)
        unique_errors, counts = np.unique(num_data_errors, return_counts=True)
        error_counts = dict(zip(unique_errors.tolist(), counts.tolist()))
    else:
        error_counts = {}

    # 3. Generate Statistics Dictionary
    stats = {}

    # We want entries even for 0 errors if it happened, or if the dict is empty
    # If no samples accepted, return empty or a default failure record
    if num_accepted == 0:
        return {}

    for k, count in error_counts.items():
        p_hat = count / num_accepted

        # Standard Error (Wald)
        std_error = math.sqrt(p_hat * (1 - p_hat) / num_accepted)

        # Wilson Interval (Better for low prob)
        wilson_low, wilson_high = calculate_wilson_interval(count, num_accepted)

        # Wald Interval (Standard, but can be inaccurate near 0)
        z = 1.96  # 95% confidence
        wald_low = p_hat - z * std_error
        wald_high = p_hat + z * std_error

        stats[k] = {
            "n": n,  # Useful to keep the context
            "k": k,  # The specific error count (Hamming distance)
            't': t,
            "count": count,
            "total_samples": total_samples_attempted,
            "num_accepted": num_accepted,
            "probability": p_hat,  # P(k | accepted)
            "acceptance_rate": acceptance_rate,
            "std_error": std_error,
            "ci_wilson_lower": wilson_low,
            "ci_wilson_upper": wilson_high,
            "ci_wald_lower": max(0.0, wald_low),
            "ci_wald_upper": min(1.0, wald_high),
        }

    return stats


def run_simulation(n: int, t: int, p: float, num_samples: int = 1_000_000, save_samples: bool = False):
    circ = cat_state_FT(n, t, run_verification=False)
    if circ is None:
        return None

    num_flags = circ.num_qubits - n
    noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=0, p_meas=2 / 3 * p)
    noisy_circ.append("M", range(num_flags, circ.num_qubits))

    # Run the simulation
    circuit_sampler = noisy_circ.compile_sampler()
    samples: np.ndarray = circuit_sampler.sample(num_samples)

    if save_samples:
        save_simulation_data(n, t, samples)

    # Process metrics
    stats = process_samples(num_samples, samples, num_flags, n, t)

    # Optional: Print summary of counts for quick debugging
    counts_summary = {k: v['count'] for k, v in stats.items()}
    print(f"Stats for {t}-FT {n}-cat (p={p}): {counts_summary}")

    return stats


def visualise_pk_per_n(collected_data, t):
    # 1. Convert flat list of dicts to DataFrame
    df = pd.DataFrame(collected_data)

    # Check if data exists
    if df.empty:
        print("No data to visualise.")
        return

    # ---------------------------------------------------------
    # 2. Filtering & Preparation
    # ---------------------------------------------------------

    # Filter for only 1 <= k <= 5
    # FIX: We filter on 'k', not 'probability'
    df_filtered = df[df['k'].between(1, 5)].copy()

    if df_filtered.empty:
        print("No data found for 1 <= k <= 5.")
        return

    # Create the label for the legend
    # FIX: We use 'k' to generate the label "k=1", "k=2", etc.
    df_filtered['k_label'] = df_filtered['k'].apply(lambda x: f"k={int(x)}")

    # Sort to ensure legend appears in order k=1, k=2, ...
    df_filtered.sort_values(by=['k', 'n'], inplace=True)

    # ---------------------------------------------------------
    # 3. Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=120)

    # Use the 'viridis' palette exactly as requested before
    sns.lineplot(
        data=df_filtered,
        x='n',
        y='probability',  # This matches the key in your new stats dict
        hue='k_label',
        style='k_label',
        markers=['o'] * 5,
        dashes=False,
        palette='viridis',
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
    # plt.savefig(f"simulation_data/P_k_per_n_at_t_{t}.png", dpi=1200)
    plt.show()


def visualise_acceptance_heatmap(collected_data):
    df = pd.DataFrame(collected_data)

    if 't' not in df.columns:
        print("Error: Column 't' missing. Please update process_simulation to save 't'.")
        return

    # Pivot the data: Rows=t, Columns=n, Values=acceptance_rate
    pivot_table = df.pivot_table(index='t', columns='n', values='acceptance_rate', aggfunc='mean')

    pivot_table.sort_index(ascending=False, inplace=True)

    plt.figure(figsize=(14, 4))

    # Plot Heatmap
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1%",
        cmap="RdYlBu",
        linewidths=0.5,
        linecolor='white',
        square=True,
        annot_kws={"size": 7},
        cbar_kws={'label': 'Acceptance Rate', 'shrink': 0.7}
    )

    # Styling
    ax.set_title("Acceptance Rate", fontsize=14)
    ax.set_xlabel("Cat State Size (n)", fontsize=12)
    ax.set_ylabel("Fault-distance (t)", fontsize=12)

    # Ensure X and Y ticks are horizontal and visible
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    ax.tick_params(left=True, bottom=True, length=5)

    plt.tight_layout()
    plt.savefig(f"simulation_data/AR_heatmap.png", dpi=400)
    plt.show()


# 1. Define a helper function that handles a SINGLE 'n'
def process_simulation(n, t, p, num_samples):
    stats_dict = run_simulation(n, t, p, num_samples)
    if stats_dict is None:
        return []
    return list(stats_dict.values())


if __name__ == "__main__":
    init_data_folder()
    start_time = time.time()

    print("Starting simulation loop...")

    t=7

    parallel_results = Parallel(n_jobs=-2)(
        delayed(process_simulation)(n, t=t, p=0.01, num_samples=10_000_000) for t in range(3, 8) for n in range(8, 36)
    )
    collected_data = [item for sublist in parallel_results for item in sublist]

    with open(f"simulation_data/simulation_results_t_{t}.json", "w") as f:
        json.dump(collected_data, f, indent=4)

    print("Simulation complete")
    visualise_acceptance_heatmap(collected_data)

    print("--- %s seconds ---" % (time.time() - start_time))
