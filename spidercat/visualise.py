import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator


def visualise_acceptance_heatmap(df):
    # Pivot the data: Rows=t, Columns=n, Values=acceptance_rate
    pivot_table = df.pivot_table(index='t', columns='n', values='acceptance_rate', aggfunc='mean')

    pivot_table.sort_index(ascending=False, inplace=True)

    # --- DYNAMIC SIZING LOGIC ---
    num_rows = len(pivot_table.index)
    num_cols = len(pivot_table.columns)

    cell_size = 0.6
    fig_width = (num_cols * cell_size)
    fig_height = (num_rows * cell_size) + (3 * cell_size)

    # Create figure with calculated dimensions
    plt.figure(figsize=(fig_width, fig_height), dpi=300)

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
    plt.savefig(f"simulation_data/AR_heatmap.png")
    plt.show()


def visualise_pk_per_n(df, t):
    # Filter for only 1 <= k <= 5
    df_filtered = df[(df['n'] >= 8) & (df['t'] == t) & df['k'].between(1, 10)].copy()

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
    plt.savefig(f"simulation_data/Pk_per_n_at_t{t}.png", dpi=1200)
    plt.show()


def visualise_pk_per_t_1(df, n):
    results = []

    grouped = df.groupby(['n', 't', 'p'])

    for (n, t, p), group in grouped:
        acc_rate = group['acceptance_rate'].iloc[0]

        # Calculate expected value: Sum(k * probability)
        # 'probability' here is P(k | accepted)
        mean_faults = (group['k'] * group['probability']).sum()

        results.append({
            'n': n,
            't': t,
            'p': p,
            'acceptance_rate': acc_rate,
            'mean_faults': mean_faults
        })

    df_summary = pd.DataFrame(results)
    df_summary['mean_faults'] = df_summary['mean_faults'].replace(0, np.nan)

    plt.figure(figsize=(10, 6), dpi=300)
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df_summary,
        x='p',
        y='mean_faults',
        hue='t',
        palette='viridis',
        marker='s',
        linewidth=2
    )
    plt.xscale('log')
    plt.title(f"Threshold Plot: Average Faults vs Physical Error @ n={n}")
    plt.ylabel(r"Average Number of Faults ($\mathbb{E}[k]$)")
    plt.xlabel("Physical Error Rate ($p$)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Error Probability", loc='upper left')
    ax2 = plt.twinx()

    sns.lineplot(
        data=df_summary,
        x='p',
        y='acceptance_rate',
        hue='t',
        palette='viridis',
        marker='X',
        linestyle=':',
        linewidth=1.5,
        alpha=0.5,
        ax=ax2,
    )
    plt.ylabel(r"Acceptance Rate")
    plt.legend(title="Acceptance Rate", loc='upper right')
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    plt.tight_layout()
    plt.savefig(f"simulation_data/EPk_per_p_at_n{n}.png")
    plt.show()


def visualise_pk_per_t_2(df, n):
    results = []

    df_filtered = df[df['k'] <= 4]
    grouped = df_filtered.groupby(['n', 't', 'p'])

    for (n, t, p), group in grouped:
        acc_rate = group['acceptance_rate'].iloc[0]

        # Calculate expected value: Sum(k * probability)
        # 'probability' here is P(k | accepted)
        mean_faults = (group['probability']).sum()

        results.append({
            'n': n,
            't': t,
            'p': p,
            'acceptance_rate': 1 - acc_rate,
            'mean_faults': mean_faults
        })

    df_summary = pd.DataFrame(results)
    df_summary['mean_faults'] = df_summary['mean_faults'].replace(0, np.nan)

    plt.figure(figsize=(10, 6), dpi=100)
    sns.lineplot(
        data=df_summary,
        x='p',
        y='mean_faults',
        hue='t',
        palette='viridis',
        marker='s',
        linewidth=2
    )
    plt.xscale('log')
    plt.title(f"Probability of 4 or less faults vs Physical Error @ n={n}")
    plt.ylabel(r"Probability of 4 or less faults ($\mathbb{P}(k \leq 4)$)")
    plt.xlabel("Physical Error Rate ($p$)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Error Probability", loc='upper left')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax2 = plt.twinx()

    sns.lineplot(
        data=df_summary,
        x='p',
        y='acceptance_rate',
        hue='t',
        palette='viridis',
        marker='X',
        linestyle=':',
        linewidth=1.5,
        alpha=0.5,
        ax=ax2,
    )
    plt.ylabel(r"Rate of Post-Selection")
    plt.legend(title="Post-Selection", loc='upper right')
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    plt.tight_layout()
    plt.savefig(f"simulation_data/k_less_4_per_p_at_n{n}.png")
    plt.show()


def visualise_method_comparison(methods_data_dict, t):
    """
    Compares multiple methods for a fixed fault distance t with Dual Axis.

    Args:
        methods_data_dict (dict): Keys represent method names, Values are data lists.
        t (int): The fault distance to filter by.
        plot_as_failure_rate (bool): If True, plots failure rate (Log Scale).
    """
    results = []

    # 1. Data Aggregation
    for method_name, raw_data in methods_data_dict.items():

        # Filter for relevant scope
        # Note: We filter n >= 8 and t == t
        if isinstance(raw_data, tuple):
            t_extra = raw_data[1]
            raw_data = raw_data[0]
            df = pd.DataFrame(raw_data)
            scope_df = df[(df['n'] >= 10) & (df['t'] == (t + t_extra))]
        else:
            df = pd.DataFrame(raw_data)
            scope_df = df[(df['n'] >= 10) & (df['t'] == t)]

        if scope_df.empty:
            print(f"Warning: No data for method '{method_name}' at t={t}")
            continue

        # Group by 'n' to calculate metrics per cat state size
        for n, group in scope_df.groupby('n'):
            # Metric 1: Probability of success (k < t)
            # Sum probability of all k where k < t
            success_prob = group[group['k'] <= t]['probability'].sum()
            if 1.0 - success_prob < 1e-8:
                continue

            # Metric 2: Acceptance Rate (Constant for a specific simulation n,t)
            # We take the mean or just the first value
            acc_rate = group['acceptance_rate'].iloc[0]
            num_flags = group['num_flags'].iloc[0]

            results.append({
                'n': n,
                'method': method_name,
                'success_prob': success_prob,
                'failure_prob': 1.0 - success_prob,
                'acceptance_rate': acc_rate,
                'num_flags': num_flags,
            })

    if not results:
        print("No valid data found to plot.")
        return

    plot_df = pd.DataFrame(results)

    # 2. Setup Plot
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=120)
    ax2 = ax1.twinx()  # Create secondary Y-axis

    # Assign distinct colors to each method
    unique_methods = plot_df['method'].unique()
    palette = sns.color_palette("bright", len(unique_methods))
    method_colors = dict(zip(unique_methods, palette))

    # 3. Plotting Loop
    for method in unique_methods:
        subset = plot_df[plot_df['method'] == method].sort_values('n')
        color = method_colors[method]

        # --- Primary Axis (Left): Probability ---
        y_val = subset['failure_prob']

        ax1.plot(
            subset['n'], y_val,
            color=color, linestyle='-', linewidth=2, marker='o',
            label=method  # Label for legend
        )

        # --- Secondary Axis (Right): Acceptance Rate ---
        ax2.plot(
            subset['n'], subset['num_flags'],
            color=color, linestyle='--', linewidth=1.5, marker='x', alpha=0.7
        )

    # 4. Styling & Legends

    # Left Axis Styling
    ax1.set_ylabel(f"Probability of $> {t}$ Faults (Failure)", fontsize=12)
    ax1.set_yscale('log')

    ax1.set_xlabel("Cat State Size (n)", fontsize=12)
    ax1.grid(True, which="both", ls="--", color='lightgrey', alpha=0.5)

    # Right Axis Styling
    # ax2.set_ylabel("Acceptance Rate", fontsize=12, rotation=270, labelpad=15)
    # ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylabel("Number of Flags", fontsize=12, rotation=270, labelpad=15)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.set_ylim(0, 1.05)  # Percents usually 0-1

    # Combined Legend Construction
    # Part A: Method Colors
    handles, labels = ax1.get_legend_handles_labels()
    legend1 = ax1.legend(handles, labels, title="Method", loc='upper left', bbox_to_anchor=(1.1, 1))
    ax1.add_artist(legend1)  # Preserve first legend

    # Part B: Line Styles (Explanation)
    style_lines = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', marker='o'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', marker='x')
    ]
    # style_labels = ['Probability (Left)', 'Acceptance Rate']
    style_labels = ['Probability (Left)', 'Number of Flags']
    ax1.legend(style_lines, style_labels, loc='upper left', bbox_to_anchor=(1.1, 0.7))

    plt.title(f"Method Comparison: Probability vs Acceptance (t={t})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"simulation_data/k_less_t_per_n_at_t{t}.png")
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    import json

    with open(f"simulation_data/simulation_results_t_n_spider-cat_ham.json", "r") as f:
        df_sc_ham = pd.DataFrame(json.load(f))
    with open(f"simulation_data/simulation_results_t_n_spider-cat_p1.json", "r") as f:
        df_sc_tree = pd.DataFrame(json.load(f))
    # with open(f"simulation_data/simulation_results_t_n_spider-cat_p2.json", "r") as f:
    #     df_sc_p2 = pd.DataFrame(json.load(f))
    # with open(f"simulation_data/simulation_results_t_n_spider-cat_p3.json", "r") as f:
    #     df_sc_p3 = pd.DataFrame(json.load(f))
    with open(f"simulation_data/simulation_results_t_n_spider-cat_p4.json", "r") as f:
        df_sc_p4 = pd.DataFrame(json.load(f))
    # with open(f"simulation_data/simulation_results_t_n_spider-cat_p5.json", "r") as f:
    #     df_sc_p4 = pd.DataFrame(json.load(f))
    # with open(f"simulation_data/simulation_results_t_n_spider-cat_p10.json", "r") as f:
    #     df_sc_p10 = pd.DataFrame(json.load(f))
    with open(f"simulation_data/simulation_results_t_n_flag-at-origin_p1.json", "r") as f:
        df_FAO = pd.DataFrame(json.load(f))
    with open(f"simulation_data/simulation_results_t_n_MQT_p1.json", "r") as f:
        df_MQT = pd.DataFrame(json.load(f))
    methods = {
        # "SpiderCat (H-Path)": df_sc_ham,
        "SpiderCat (Tree)": df_sc_tree,
        "SpiderCat (Tree T+1)": (df_sc_tree, 1),
        # "SpiderCat (Tree T+2)": (df_sc_tree, 2),
        # "SpiderCat (Tree T+3)": (df_sc_tree, 3),
        # "SpiderCat (3-Forest)": df_sc_p3,
        "SpiderCat (4-Forest)": df_sc_p4,
        # "SpiderCat (5-Forest)": df_sc_p5,
        # "SpiderCat (2-Path)": df_sc_p2,
        # "SpiderCat (3-Path)": df_sc_p3,
        # "SpiderCat (4 forest)": df_sc_p4,
        # "SpiderCat (10 forest)": df_sc_p10,
        "Flag at Origin": df_FAO,
        "MQT": df_MQT
    }
    # visualise_method_comparison(methods, t=1)
    # visualise_method_comparison(methods, t=2)
    visualise_method_comparison(methods, t=3)
    visualise_method_comparison(methods, t=4)
    visualise_method_comparison(methods, t=5)
    # visualise_method_comparison(methods, t=6)

    # with open(f"simulation_data/simulation_results_t_n.json", "r") as f:
    #     collected_data = json.load(f)
    # df_t_n = pd.DataFrame(collected_data)
    #
    # visualise_acceptance_heatmap(df_sc_p1)
    # for t in [3]:
    #     visualise_pk_per_n(df_sc_p1, t)

    # for n in [24]:
    #     with open(f"simulation_data/simulation_results_t_p_n{n}.json", "r") as f:
    #         collected_data = json.load(f)
    #     df_t_p = pd.DataFrame(collected_data)
    #     visualise_pk_per_t_1(df_t_p, n)
    #     visualise_pk_per_t_2(df_t_p, n)
