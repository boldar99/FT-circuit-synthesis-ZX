import json
import math
import time
from pathlib import Path

import numpy as np
import stim
from joblib import Parallel, delayed
from scipy.stats import norm

from cat_state.circuit_extraction import make_stim_circ_noisy
from cat_state.visalise import visualise_acceptance_heatmap, visualise_pk_per_n

cwd = Path.cwd()


def init_data_folder():
    Path(f"{cwd}/simulation_data").mkdir(parents=True, exist_ok=True)


def load_stim_circuit(t: int, n: int):
    my_file = Path(f"{cwd}/circuits/cat_state_t{t}_n{n}.stim")
    if my_file.is_file():
        return stim.Circuit(my_file.read_text())
    return None


def save_simulation_data(n, t, samples: np.ndarray):
    # now = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = f"simulation_data/samples_{n}_{t}.npz"
    stim.write_shot_data_file(data=samples, path=file_name, format="b8")
    # np.savez_compressed(file_name, samples)
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


def process_samples(samples: np.ndarray, num_flags: int, n: int, t: int, p: float, total_samples_attempted: int):
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
            "n": n,
            't': t,
            'p': p,
            "k": k,
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
    circ = load_stim_circuit(t, n)
    if circ is None:
        return None

    num_flags = circ.num_qubits - n
    # noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=0, p_meas=2 / 3 * p)
    noisy_circ = make_stim_circ_noisy(circ, p_2=p, p_init=2 / 3 * p, p_meas=2 / 3 * p, p_mem=0)
    noisy_circ.append("M", range(num_flags, circ.num_qubits))

    # Run the simulation
    circuit_sampler = noisy_circ.compile_sampler()
    samples: np.ndarray = circuit_sampler.sample(num_samples)

    if save_samples:
        save_simulation_data(n, t, samples)

    # Process metrics
    stats = process_samples(samples, num_flags, n, t, p, num_samples)

    # Optional: Print summary of counts for quick debugging
    counts_summary = {k: v['count'] for k, v in stats.items()}
    print(f"Stats for {t}-FT {n}-cat (p={p}): {counts_summary}")

    return stats


# 1. Define a helper function that handles a SINGLE 'n'
def process_simulation(n, t, p, num_samples):
    stats_dict = run_simulation(n, t, p, num_samples)
    if stats_dict is None:
        return []
    return list(stats_dict.values())


def simulate_t_n(ts, ns):
    print("Starting simulation loop, varying values of t and n")
    parallel_results = Parallel(n_jobs=-2)(
        delayed(process_simulation)(n, t, p=0.01, num_samples=100_000) for t in ts for n in ns
    )
    collected_data = [item for sublist in parallel_results for item in sublist]
    with open(f"simulation_data/simulation_results_t_n.json", "w") as f:
        json.dump(collected_data, f, indent=4)
    print("Simulation complete")
    print()


def simulate_t_p(ts, ps, n):
    print("Starting simulation loop, varying values of t and p")
    parallel_results = Parallel(n_jobs=-2)(
        delayed(process_simulation)(n=n, t=t, p=p, num_samples=100_000) for t in ts for p in ps
    )
    collected_data = [item for sublist in parallel_results for item in sublist]
    with open(f"simulation_data/simulation_results_t_p_n{n}.json", "w") as f:
        json.dump(collected_data, f, indent=4)
    print("Simulation complete")
    print()


if __name__ == "__main__":
    init_data_folder()
    start_time = time.time()

    # simulate_t_n(range(1, 8), range(8, 101))
    # simulate_t_p(range(1, 8), (10 ** np.linspace(-0.5, -3, 26)).tolist(), n=24)
    # simulate_t_p(range(1, 8), (10 ** np.linspace(-0.5, -3, 26)).tolist(), n=34)
    # simulate_t_p(range(1, 8), (10 ** np.linspace(-0.5, -3, 26)).tolist(), n=50)
    simulate_t_p(range(1, 6), (10 ** np.linspace(-0.5, -3, 26)).tolist(), n=80)


    print("--- %s seconds ---" % (time.time() - start_time))
