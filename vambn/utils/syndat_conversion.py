from pathlib import Path
import pandas as pd
import typer


def main(
    original_input: Path,
    decoded_input: Path,
    synthetic_input: Path,
    original_output: Path,
    decoded_output: Path,
    synthetic_output: Path,
) -> None:
    """
    Processes and pivots the original, decoded, and synthetic CSV files, then saves the results.

    Args:
        original_input (Path): Path to the original input CSV file.
        decoded_input (Path): Path to the decoded input CSV file.
        synthetic_input (Path): Path to the synthetic input CSV file.
        original_output (Path): Path where the processed original output CSV file will be saved.
        decoded_output (Path): Path where the processed decoded output CSV file will be saved.
        synthetic_output (Path): Path where the processed synthetic output CSV file will be saved.
    """
    # Read input CSV files
    synthetic = pd.read_csv(synthetic_input)
    decoded = pd.read_csv(decoded_input)
    original = pd.read_csv(original_input)

    # Order by SUBJID and VISIT
    synthetic = synthetic.sort_values(["SUBJID", "VISIT"])
    decoded = decoded.sort_values(["SUBJID", "VISIT"])
    original = original.sort_values(["SUBJID", "VISIT"])

    # Get the common columns of synthetic and decoded
    common_cols = synthetic.columns.intersection(decoded.columns)
    # Subset all dataframes to the common columns
    synthetic = synthetic[common_cols]
    decoded = decoded[common_cols]
    original = original[common_cols]

    def pivot_to_wide(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivots a dataframe to wide format by SUBJID and VISIT.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The pivoted dataframe.
        """
        pivot_df = df.pivot(index="SUBJID", columns="VISIT")
        # Add _VIS[visit_number] to the column names
        pivot_df.columns = [
            f"{col[0]}_VIS{str(col[1])}" for col in pivot_df.columns.values
        ]
        # Drop SUBJID and potentially VISIT
        pivot_df = pivot_df.reset_index().drop("SUBJID", axis=1)
        return pivot_df

    # Pivot the dataframes to wide format
    synthetic = pivot_to_wide(synthetic)
    decoded = pivot_to_wide(decoded)
    original = pivot_to_wide(original)

    # Save the processed data
    synthetic.to_csv(synthetic_output, index=False)
    decoded.to_csv(decoded_output, index=False)
    original.to_csv(original_output, index=False)


if __name__ == "__main__":
    typer.run(main)
