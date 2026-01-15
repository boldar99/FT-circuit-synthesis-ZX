import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    df_filtered = df[(df['n'] >= 8) & (df['t'] == t) & df['k'].between(1, 5)].copy()

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


def visualise_acceptance_heatmap(collected_data):
    df = pd.DataFrame(collected_data)

    if 't' not in df.columns:
        print("Error: Column 't' missing. Please update process_simulation to save 't'.")
        return

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


if __name__ == '__main__':
    import json
    with open(f"simulation_data/simulation_results_t3-t7_n2-n100.json", "r") as f:
        collected_data = json.load(f)

    visualise_acceptance_heatmap(collected_data)
    visualise_pk_per_n(collected_data, 3)
    visualise_pk_per_n(collected_data, 5)


