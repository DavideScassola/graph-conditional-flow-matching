import pandas as pd

def generate_latex_table(csv_file, output_file):
    # Load the CSV file
    df = pd.read_csv(csv_file, skiprows=2)  # Skip the first two rows of metadata
    df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names

    # Debug: Print the original DataFrame
    print("Original DataFrame:")
    print(df)

    # Rename columns if necessary
    df.rename(columns={"Unnamed: 2": "mean", "Unnamed: 3": "std"}, inplace=True)

    # Pivot the table to make datasets columns and methods rows for mean and std
    pivot_mean = df.pivot(index="method_name", columns="dataset", values="mean")
    pivot_std = df.pivot(index="method_name", columns="dataset", values="std")

    # Debug: Print the pivot tables
    print("Pivot Table (Mean):")
    print(pivot_mean)
    print("Pivot Table (Std):")
    print(pivot_std)

    # Dynamically generate column headers based on the datasets in the pivot table
    datasets = sorted(pivot_mean.columns.tolist(), key=str.lower)  # Case-insensitive sort
    column_headers = " & ".join([f"\\textbf{{{dataset}}}" for dataset in datasets])

    # Reorder the pivot tables to match the sorted dataset order
    pivot_mean = pivot_mean[datasets]
    pivot_std = pivot_std[datasets]

    # Correctly define the column structure for the tabular environment
    column_structure = "l" + "c" * len(datasets)

    # Define the LaTeX table structure
    latex_table = """
\\begin{{table}}[h]
\\centering
\\caption{{Average accuracy of an XGBoost multi-table discriminator using rows with aggregated statistics. For the CORA experiment, after generating the data we removed duplicated categories for the same article. Without this step, the average performance would be $0.61$.}}
\\label{{tab:results}}
\\begin{{tabular}}{{{column_structure}}}
\\toprule
 & {column_headers} \\\\
\\midrule
""".format(column_structure=column_structure, column_headers=column_headers)

    # Find the minimum value for each column
    min_values = pivot_mean.min(axis=0, skipna=True)

    # Add rows for each method
    for method, row_mean in pivot_mean.iterrows():
        row_std = pivot_std.loc[method]  # Get the corresponding std row
        # Debug: Print each row being added
        print(f"Method: {method}, Row Mean: {row_mean}, Row Std: {row_std}")

        latex_row = f"{method} & " + " & ".join(
            [
                f"$\\approx 1$" if mean >= 0.99 else
                f"$\\mathbf{{{mean:.2f} \\pm {std:.1g}}}$" if mean == min_values[col] and pd.notna(mean) else
                f"${mean:.2f} \\pm {std:.1g}$" if pd.notna(mean) else "--"
                for col, (mean, std) in zip(row_mean.index, zip(row_mean, row_std))
            ]
        ) + r" \\"
        latex_table += latex_row + "\n"

    # Close the LaTeX table
    latex_table += """
\\bottomrule
\\end{tabular}
\\end{table}
"""

    # Save the LaTeX table to a file
    with open(output_file, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to {output_file}")


# Example usage
csv_file = "pivot_table.csv"  # Path to your CSV file
output_file = "results_table.tex"  # Path to save the LaTeX table
generate_latex_table(csv_file, output_file)