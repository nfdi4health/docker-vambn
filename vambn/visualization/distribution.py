import json
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score

from vambn.metrics.jensen_shannon import jensen_shannon_distance
from vambn.utils.helpers import handle_nan_values

app = typer.Typer()


def remove_vis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove '_VIS1' from column names in the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with modified column names.
    """
    data.columns = [x.replace("_VIS1", "") for x in data.columns]
    return data


def pad_with_nan(x: List[Any], num: int) -> List[Any]:
    """
    Pad a list with NaN values to a specified length.

    Args:
        x (List[Any]): The input list.
        num (int): The desired length of the list.

    Returns:
        List[Any]: The padded list.
    """
    return x + [np.nan] * (num - len(x))


@app.command()
def compare_data(
    original_file: Path,
    decoded_file: Path,
    virtual_file: Path,
    output_data: Path,
    output_data_dec: Path,
    metric_file: Path,
    grouping: Path,
    dataset_name: str,
    var: str,
) -> None:
    """
    Compare data from original, decoded, and virtual sources, and generate metrics and visualizations.

    Args:
        original_file (Path): Path to the original data CSV file.
        decoded_file (Path): Path to the decoded data CSV file.
        virtual_file (Path): Path to the virtual data CSV file.
        output_data (Path): Path to save the PDF report of all data comparisons.
        output_data_dec (Path): Path to save the PDF report of original vs. decoded comparisons.
        metric_file (Path): Path to save the JSON file with Jensen-Shannon distances.
        grouping (Path): Path to the grouping CSV file.
        dataset_name (str): Name of the dataset.
        var (str): Experiment variable name.
    """
    groups = pd.read_csv(grouping)
    # Read in the data
    decoded_data = remove_vis(pd.read_csv(decoded_file))
    virtual_data = pd.read_csv(virtual_file)

    # Read original data
    original_data = pd.read_csv(original_file, index_col=0).reset_index()

    # Reduce dataframes to overlapping columns
    common_cols = list(
        set(original_data.columns)
        & set(decoded_data.columns)
        & set(virtual_data.columns)
    )
    original_data = original_data[common_cols]
    decoded_data = decoded_data[common_cols]
    virtual_data = virtual_data[common_cols]

    # Clip inf values to 1e20 and -1e20
    decoded_data = decoded_data.replace([np.inf, -np.inf], [1e20, -1e20])
    virtual_data = virtual_data.replace([np.inf, -np.inf], [1e20, -1e20])

    jsd_list = []

    output_data.parent.mkdir(exist_ok=True, parents=True)
    # Open the pdf file
    all_folder = output_data.parent / "all_data"
    all_folder.mkdir(exist_ok=True, parents=True)
    with PdfPages(output_data) as pdf:
        for col in common_cols:
            if col in ["SUBJID", "VISIT"]:
                continue

            dtype = groups.loc[
                groups["column_names"] == col, "hivae_types"
            ].tolist()[0]
            plt.figure()
            sub_orig = (
                original_data.reset_index()
                .loc[:, ["SUBJID", "VISIT", col]]
                .dropna()
            )
            max_visit = sub_orig["VISIT"].max()
            orig = sub_orig[col].tolist()
            dec = decoded_data.reset_index().loc[
                decoded_data["VISIT"] <= max_visit, ["SUBJID", "VISIT", col]
            ]
            dec.rename(columns={col: f"{col}_dec"}, inplace=True)
            dec = (
                pd.merge(sub_orig, dec, on=["SUBJID", "VISIT"], how="left")
                .drop(columns=col)[f"{col}_dec"]
                .tolist()
            )
            vir = virtual_data.loc[
                virtual_data["VISIT"] <= max_visit, col
            ].tolist()

            # Identify outliers
            lower_bound = min(orig) * 0.5 if min(orig) < 0 else min(orig) * 1.5
            if lower_bound == 0:
                lower_bound = -0.5

            max_length = max(len(orig), len(dec), len(vir))
            orig = pad_with_nan(orig, max_length)
            dec = pad_with_nan(dec, max_length)
            vir = pad_with_nan(vir, max_length)

            wide = pd.DataFrame(
                {
                    "Real (original)": orig,
                    "Decoded": dec,
                    "Virtual": vir,
                }
            )
            plot_df = wide.melt().rename(
                columns={"variable": "type", "value": "value"}
            )

            dec_jsd = round(jensen_shannon_distance(orig, dec, dtype), 3)
            vir_jsd = round(jensen_shannon_distance(orig, vir, dtype), 3)

            jsd_list.append(
                {
                    "column": col,
                    "jsd_decoded": dec_jsd,
                    "jsd_virtual": vir_jsd,
                    "dataset_var": dataset_name,
                    "experiment": var,
                }
            )

            title = (
                f"Distribution for {col}\n Decoded {dec_jsd}; Virtual {vir_jsd}"
            )

            plt.figure()
            if dtype in ["cat", "categorical", "count"]:
                plot_df_freq = (
                    plot_df.groupby("type")["value"]
                    .value_counts(normalize=True)
                    .rename("frequency")
                    .reset_index()
                )

                # Plot using sns.barplot
                sns.barplot(
                    x="value",
                    y="frequency",
                    hue="type",
                    data=plot_df_freq,
                    dodge=True,
                )
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.title(title)
                plt.legend()

            else:
                print(
                    f"Plotting {col}, min value in plot_df: {plot_df['value'].min()}, max value in plot_df: {plot_df['value'].max()}"
                )
                plt.subplot(211)
                sns.violinplot(x="type", y="value", data=plot_df.dropna())
                plt.ylabel("Value")
                plt.title(title)
                plt.legend()

                plt.subplot(212)
                plt.axis("off")
                plt.table(
                    cellText=wide.describe().round(1).values,
                    colLabels=wide.columns,
                    rowLabels=wide.describe().index,
                    loc="center",
                )

            plt.tight_layout()

            pdf.savefig()
            plt.savefig(all_folder / f"{col}.png", dpi=300)

            plt.close()

    # Open the pdf file
    indiv_folder = output_data_dec.parent / "original_vs_decoded"
    indiv_folder.mkdir(exist_ok=True, parents=True)
    with PdfPages(output_data_dec) as pdf:
        for col in common_cols:
            if col in ["SUBJID", "VISIT"]:
                continue

            dtype = groups.loc[
                groups["column_names"] == col, "hivae_types"
            ].tolist()[0]
            sub_orig = (
                original_data.reset_index()
                .loc[:, ["SUBJID", "VISIT", col]]
                .dropna()
            )
            max_visit = sub_orig["VISIT"].max()
            orig = sub_orig[col].tolist()
            dec = decoded_data.reset_index().loc[
                decoded_data["VISIT"] <= max_visit, ["SUBJID", "VISIT", col]
            ]
            dec.rename(columns={col: f"{col}_dec"}, inplace=True)
            dec = (
                pd.merge(sub_orig, dec, on=["SUBJID", "VISIT"], how="left")
                .drop(columns=col)[f"{col}_dec"]
                .tolist()
            )

            # Identify outliers
            lower_bound = min(orig) * 0.5 if min(orig) < 0 else min(orig) * 1.5
            if lower_bound == 0:
                lower_bound = -0.5

            max_length = max(len(orig), len(dec))
            orig = pad_with_nan(orig, max_length)
            dec = pad_with_nan(dec, max_length)

            plt.figure()

            wide = pd.DataFrame(
                {
                    "Real (original)": orig,
                    "Decoded": dec,
                }
            )
            plot_df = wide.melt().rename(
                columns={"variable": "type", "value": "value"}
            )

            dec_jsd = round(jensen_shannon_distance(orig, dec, dtype), 3)
            try:
                orig, dec = handle_nan_values(orig, dec)
                orig = orig.iloc[:, 0].to_numpy()
                dec = dec.iloc[:, 0].to_numpy()

                if np.isinf(dec).any():
                    # replace inf with 1e20 and -inf with -1e20
                    dec = dec.replace([np.inf, -np.inf], [1e20, -1e20])
                if dtype in ("pos", "real", "count", "truncate_norm", "gamma"):
                    corr, pval = pearsonr(orig, dec)
                    title = f"Distribution for {col}\n JSD: {dec_jsd}; Correlation: {round(corr, 3)} / {round(pval, 4)}, type: {dtype}"
                elif dtype == "cat" or dtype == "categorical":
                    # calculate accuracy for categorical data
                    acc = accuracy_score(orig, dec)
                    title = f"Distribution for {col}\n JSD: {dec_jsd}; Accuracy: {round(acc, 3)}, type: {dtype}"
                else:
                    raise Exception(f"Unknown dtype: {dtype}")
            except ValueError:
                raise ValueError(
                    f"Orig: {len(orig)}, Dec: {len(dec)}, any nan? {pd.isna(orig).any()} {pd.isna(dec).any()}"
                )

            if dtype in ["cat", "categorical", "count"]:
                plot_df_freq = (
                    plot_df.groupby("type")["value"]
                    .value_counts(normalize=True)
                    .rename("frequency")
                    .reset_index()
                )

                # Plot using sns.barplot
                ax = sns.barplot(
                    x="value", y="frequency", hue="type", data=plot_df_freq
                )
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.title(title)
                plt.legend()
            else:
                # make a subfigure for both the violin plot and the table
                ax = plt.subplot(211)

                # use the first subplot for the violin plot
                ax = sns.violinplot(x="type", y="value", data=plot_df)
                plt.ylabel("Value")
                plt.title(title)
                plt.legend()

                # use the second subplot for the table
                ax = plt.subplot(212)
                ax.axis("off")
                ax.table(
                    cellText=wide.describe().round(1).values,
                    colLabels=wide.columns,
                    rowLabels=wide.describe().index,
                    loc="center",
                )

            plt.tight_layout()
            plt.savefig(indiv_folder / f"{col}.png", dpi=300)
            pdf.savefig()
            plt.close()

    metric_file.parent.mkdir(exist_ok=True, parents=True)
    with metric_file.open("w+") as f:
        f.write(json.dumps(jsd_list, indent=4))


if __name__ == "__main__":
    app()
