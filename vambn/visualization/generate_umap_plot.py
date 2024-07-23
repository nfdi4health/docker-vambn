from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import typer
import umap

from vambn.utils.helpers import encode_numerical_columns, handle_nan_values


def generate_umap_plot(
    grouping_file: Path,
    real_data: Path,
    decoded_data: Path,
    virtual_data: Path,
    output_decoded: Path,
    output_virtual: Path,
    max_samples: int = 1000,
):
    """
    Generate UMAP plots comparing real, decoded, and virtual data.

    Args:
        grouping_file (Path): The path to the CSV file containing column name mappings.
        real_data (Path): The path to the CSV file containing real data.
        decoded_data (Path): The path to the CSV file containing decoded data.
        virtual_data (Path): The path to the CSV file containing virtual data.
        output_decoded (Path): The path to the output file for the decoded UMAP plot.
        output_virtual (Path): The path to the output file for the virtual UMAP plot.
        max_samples (int): The maximum number of samples to use for the UMAP plot.
    """
    grouping = pd.read_csv(grouping_file)
    real = pd.read_csv(real_data)
    decoded = pd.read_csv(decoded_data)
    virtual = pd.read_csv(virtual_data)

    def _prepare_data(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Prepare the data by encoding numerical columns and sorting by subject and visit.

        Args:
            df (pd.DataFrame): The dataframe to prepare.
            cols (List[str]): List of relevant columns to keep.

        Returns:
            pd.DataFrame: The prepared dataframe.
        """
        x = df.loc[df["VISIT"] == 1, cols]
        x = encode_numerical_columns(x)
        if "subjid" in x.columns:
            x = x.rename(columns={"subjid": "SUBJID"})
        x.sort_values(by=["SUBJID", "VISIT"], inplace=True)
        return x

    # Derive relevant columns from grouping file
    subset_without_stalone = grouping.loc[
        ~grouping["technical_group_name"].str.startswith("stalone"), :
    ]
    relevant_columns = subset_without_stalone["column_names"].tolist()

    # Filter relevant columns from real and synthetic data
    available_columns = [
        col
        for col in relevant_columns
        if col in real.columns
        and col in virtual.columns
        and col in decoded.columns
    ] + ["SUBJID", "VISIT"]
    real = _prepare_data(real, available_columns)
    decoded = _prepare_data(decoded, available_columns)
    virtual = _prepare_data(virtual, available_columns)
    assert real.shape[1] == decoded.shape[1] == virtual.shape[1]

    real_1, decoded = handle_nan_values(real, decoded)
    real_2, virtual = handle_nan_values(real, virtual)

    # Drop SUBJID and VISIT columns
    real_1 = real_1.drop(columns=["SUBJID", "VISIT"])
    real_2 = real_2.drop(columns=["SUBJID", "VISIT"])
    decoded = decoded.drop(columns=["SUBJID", "VISIT"])
    virtual = virtual.drop(columns=["SUBJID", "VISIT"])

    # Sample data if necessary
    if real_1.shape[0] > max_samples:
        real_1 = real_1.sample(n=max_samples, random_state=42)
    if real_2.shape[0] > max_samples:
        real_2 = real_2.sample(n=max_samples, random_state=42)
    if decoded.shape[0] > max_samples:
        decoded = decoded.sample(n=max_samples, random_state=42)
    if virtual.shape[0] > max_samples:
        virtual = virtual.sample(n=max_samples, random_state=42)

    generate_plot(output_decoded, real_1, decoded)
    generate_plot(output_virtual, real_2, virtual)

    print("Done!")


def generate_plot(output_file, real_data, synthetic_data):
    """
    Generate and save UMAP plot data.

    Args:
        output_file (Path): The path to the output file for the UMAP plot.
        real_data (pd.DataFrame): Dataframe containing real data.
        synthetic_data (pd.DataFrame): Dataframe containing synthetic data.
    """
    concat = pd.concat([real_data, synthetic_data]).clip(-1e15, 1e15)
    reducer = umap.UMAP(n_components=2, random_state=42)
    dim_reduct = reducer.fit_transform(concat)
    border = real_data.shape[0]
    x_real = dim_reduct[:border, 0]
    y_real = dim_reduct[:border, 1]
    x_virtual = dim_reduct[border:, 0]
    y_virtual = dim_reduct[border:, 1]

    fig, ax = plt.subplots()
    ax.scatter(x_real, y_real, c="#d62728", label="real", alpha=0.35, s=5)
    ax.scatter(
        x_virtual, y_virtual, c="#17becf", label="virtual", alpha=0.35, s=5
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    typer.run(generate_umap_plot)
