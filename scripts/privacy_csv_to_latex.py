import pandas as pd

def generate_latex_table(input_csv, output_tex):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Group by dataset and table, and compute mean and std of the mean column
    agg_df = df.groupby(["dataset", "table", "n_rows", "n_cols"]).agg(
        mean_synth=("mean_synth_below_quantile_percentage", "mean"),
        std_synth=("mean_synth_below_quantile_percentage", "std"),
        mean_dcr=("mean_dcr_privacy_score", "mean"),
        std_dcr=("mean_dcr_privacy_score", "std")
    ).reset_index()

    # Start building the LaTeX table
    latex_table = r"""
\begin{table}[ht]
\centering
\caption{Privacy results showing the below quantile percentage and privacy score for each dataset and table. The results are presented as mean $\pm$ standard deviation.}
\label{tab:privacy_results}

\begin{tabular}{l l r r l l}
\toprule
\textbf{Dataset} & \textbf{Table} & \textbf{\# Rows} & \textbf{\# Features} & \textbf{Below Quantile Percentage} & \textbf{Privacy Score} \\
\cmidrule(lr){1-6}
"""

    # Map datasets to LaTeX-friendly names
    dataset_map = {
        "airbnb-simplified_subsampled": "AirBnB",
        "Biodegradability_v1": "Biodegradability",
        "CORA_v1": "CORA",
        "imdb_MovieLens_v1": "IMDB MovieLens",
        "rossmann_subsampled": "Rossmann",
        "walmart_subsampled": "Walmart"
    }

    # Group by dataset to calculate the number of rows for each dataset
    grouped = agg_df.groupby("dataset")
    for dataset, group in grouped:
        dataset_name = dataset_map.get(dataset, dataset)
        num_rows = len(group)

        # Add the dataset row with \multirow
        first_row = True
        for _, row in group.iterrows():
            table_name = row["table"]
            n_rows = f"{int(row['n_rows']):,}"  # Format with commas
            n_cols = row["n_cols"]
            below_quantile = f"${row['mean_synth']:.2f} \\pm {row['std_synth']:.2f}$"
            privacy_score = f"${row['mean_dcr']:.2f} \\pm {row['std_dcr']:.2f}$"

            if first_row:
                latex_table += f"\multirow{{{num_rows}}}{{*}}{{{dataset_name}}} & {table_name} & {n_rows} & {n_cols} & {below_quantile} & {privacy_score} \\\\\n"
                first_row = False
            else:
                latex_table += f" & {table_name} & {n_rows} & {n_cols} & {below_quantile} & {privacy_score} \\\\\n"

        # Add a horizontal rule after each dataset group
        latex_table += r"\cmidrule(lr){1-6}" + "\n"

    # Close the LaTeX table
    latex_table += r"""
\bottomrule
\end{tabular}

\end{table}
"""

    # Save the LaTeX table to a file
    with open(output_tex, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {output_tex}")


# Example usage
if __name__ == "__main__":
    input_csv = "privacy_results_quantile_0.02_aggregated.csv"  # Input CSV file
    output_tex = "privacy_results_table.tex"  # Output LaTeX file
    generate_latex_table(input_csv, output_tex)