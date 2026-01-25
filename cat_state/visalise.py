import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


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
            'acceptance_rate': 1-acc_rate,
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


if __name__ == '__main__':
    import json

    # with open(f"simulation_data/simulation_results_t_n.json", "r") as f:
    #     collected_data = json.load(f)
    # df_t_n = pd.DataFrame(collected_data)
    #
    # visualise_acceptance_heatmap(df_t_n)
    # for t in range(1, 8):
    #     visualise_pk_per_n(df_t_n, t)

    for n in [24, 34, 50, 80]:
        with open(f"simulation_data/simulation_results_t_p_n{n}.json", "r") as f:
            collected_data = json.load(f)
        df_t_p = pd.DataFrame(collected_data)
        visualise_pk_per_t_1(df_t_p, n)
        visualise_pk_per_t_2(df_t_p, n)
