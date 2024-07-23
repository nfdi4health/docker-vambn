import json
import math
from functools import reduce
from pathlib import Path
from typing import List

import pandas as pd
import typer
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_bar,
    geom_boxplot,
    ggplot,
    labs,
    theme,
    theme_bw,
    ylim,
)
from tqdm import tqdm

app = typer.Typer()


@app.command()
def merge_results(
    grouping_file: Path,
    input_files: List[Path],
    output_csv: Path,
    output_plot: Path,
):
    """
    Merge results from multiple JSON and CSV files and generate summary plots.

    Args:
        grouping_file (Path): The path to the CSV file containing column name mappings.
        input_files (List[Path]): A list of input JSON and CSV files to process.
        output_csv (Path): The path to the output CSV file where merged results will be saved.
        output_plot (Path): The path to the output plot file where the generated plot will be saved.

    Raises:
        ValueError: If there are no input files or if a column has multiple mappings.
        Exception: If JSON files contain unsupported data types or lists.
    """
    json_files = [x for x in input_files if x.suffix == ".json"]
    csv_files = [x for x in input_files if x.suffix == ".csv"]
    grouping = pd.read_csv(grouping_file)
    # Derive mapping from grouping file
    colname_map = {
        column: grouping.loc[
            grouping["column_names"] == column, "technical_group_name"
        ]
        .drop_duplicates()
        .tolist()
        for column in grouping["column_names"]
    }
    for key, value in colname_map.items():
        if len(value) > 1:
            raise ValueError(f"Column {key} has multiple mappings {value}")
        else:
            colname_map[key] = value[0]

    if len(json_files) == 0 and len(csv_files) == 0:
        raise ValueError("No input files found")

    corr_metrics = []
    jsd_metrics = []
    for file in tqdm(json_files, desc="Reading JSON files"):
        with file.open("r") as f:
            obj = json.loads(f.read())
            if file.stem == "jsd_metrics":
                tmp = pd.DataFrame(obj)
                tmp.rename(columns={"dataset_var": "dataset"}, inplace=True)
                if "modular" in file.parent.stem:
                    tmp["variant"] = "-".join(file.parent.stem.split("_")[:2])
                else:
                    tmp["variant"] = file.parent.stem.split("_")[0]

                tmp["module"] = tmp["column"].map(lambda x: colname_map[x])

                jsd_metrics.append(tmp)
            elif file.stem == "corr_metrics":
                if "modular" in file.parent.stem:
                    obj["variant"] = "-".join(file.parent.stem.split("_")[:2])
                else:
                    obj["variant"] = file.parent.stem.split("_")[0]

                # Ensure that the values are floats and not list
                for key, value in obj.items():
                    if isinstance(value, list) and len(value) > 1:
                        raise Exception(
                            "The JSON file contains lists, which is not supported"
                        )
                    elif isinstance(value, list) and len(value) == 1:
                        obj[key] = value[0]
                    elif isinstance(value, float):
                        pass
                    elif isinstance(value, str):
                        pass
                    else:
                        raise Exception(
                            f"Unsupported type {type(value)} for key {key}"
                        )

                corr_metrics.append(obj)
            else:
                raise ValueError(f"Unknown JSON file {file.stem}")

    corr_df = pd.DataFrame(corr_metrics)
    # Reduce the list of jsd dataframes
    jsd_df = reduce(lambda x, y: pd.concat([x, y]), jsd_metrics)
    agg_jsd = (
        jsd_df.groupby(["dataset", "variant", "experiment", "module"])
        .aggregate(
            {"jsd_virtual": ["mean", "std"], "jsd_decoded": ["mean", "std"]}
        )
        .reset_index()
    )
    # agg jsd has two levels of columns, we need to flatten it
    agg_jsd.columns = ["_".join(x).strip("_") for x in agg_jsd.columns.ravel()]

    csv_objects = []
    for file in tqdm(csv_files, desc="Reading CSV files"):
        tmp = pd.read_csv(file)
        dataset_name = file.parent.stem.split("_")[-3]
        if "modular" in file.parent.stem:
            variant = "-".join(file.parent.stem.split("_")[:2])
        else:
            variant = file.parent.stem.split("_")[0]

        tmp["experiment"] = "_".join(file.parent.stem.split("_")[-2:])

        tmp["dataset"] = dataset_name
        tmp["variant"] = variant
        csv_objects.append(tmp)

    auc_df = reduce(lambda x, y: pd.concat([x, y]), csv_objects)
    module_aucs = auc_df.loc[
        auc_df["module"] != "all-modules-baseline", :
    ].drop(columns=["pauc_virVdec", "n_virtual", "n_decoded", "n_virVdec"])
    auc_df = auc_df.loc[auc_df["module"] == "all-modules-baseline", :].drop(
        columns=["pauc_virVdec", "n_virtual", "n_decoded", "n_virVdec"]
    )
    # Aggregate results
    reshaped = agg_jsd.melt(
        id_vars=["dataset", "variant", "module", "experiment"],
        var_name="metric",
        value_name="value",
    )
    merged_auc_jsd = pd.concat(
        [
            reshaped.loc[
                reshaped["metric"].isin(
                    ["jsd_virtual_mean", "jsd_decoded_mean"]
                ),
                :,
            ],
            module_aucs.melt(
                id_vars=["dataset", "variant", "module", "experiment"],
                var_name="metric",
                value_name="value",
            ),
        ]
    )

    aggregate_over_modules = (
        reshaped.groupby(["dataset", "variant", "experiment", "metric"])
        .aggregate({"value": ["mean", "std"]})
        .reset_index()
    )
    auc_df_reshaped = auc_df.drop(columns="module").melt(
        id_vars=["dataset", "variant", "experiment"],
        var_name="metric",
        value_name="value",
    )

    aggregate_over_modules.columns = [
        "_".join(x).strip("_") for x in aggregate_over_modules.columns.ravel()
    ]
    aggregate_over_modules.rename(
        columns={"value_mean": "value", "value_std": "std"}, inplace=True
    )

    aggregate_over_modules = pd.concat(
        [aggregate_over_modules, auc_df_reshaped]
    )

    corr_df_reshaped = corr_df.melt(
        id_vars=["dataset", "variant", "experiment"],
        var_name="metric",
        value_name="value",
    )

    # Merge the two dataframes
    merged = pd.concat([corr_df_reshaped, aggregate_over_modules])

    def _assign_type(x: str) -> str:
        if "pearson_relcorr" in x:
            return "pearson-corr"
        elif "spearman_relcorr" in x:
            return "spearman-corr"
        elif "jsd" in x:
            return "jsd"
        elif "auc" in x:
            return "auc"
        else:
            raise ValueError(f"Unknown metric type {x}")

    merged_auc_jsd["metric_type"] = merged_auc_jsd["metric"].map(_assign_type)
    merged["metric_type"] = merged["metric"].map(_assign_type)
    # Drop metrics with "_std" suffix
    merged = merged[~merged["metric"].str.contains("_std")]

    merged["data_type"] = merged["metric"].map(
        lambda x: "virtual" if "virtual" in x else "decoded"
    )
    merged_auc_jsd["data_type"] = merged_auc_jsd["metric"].map(
        lambda x: "virtual" if "virtual" in x else "decoded"
    )

    # Normalize the metrics
    def _normalize(x: pd.Series) -> pd.Series:
        if pd.isna(x["value"]) or x["value"] == "NA" or x["value"] == "NaN":
            return math.nan
        elif (
            x["metric_type"] == "pearson-corr"
            or x["metric_type"] == "spearman-corr"
        ):
            x_mod = min(x["value"], 1)
            return math.floor((1 - x_mod) * 100)
        elif x["metric_type"] == "jsd":
            return math.floor((1 - x["value"]) * 100)
        elif x["metric_type"] == "auc":
            auc = x["value"]
            if auc < 0.5:
                auc = 1 - auc

            return max(math.floor((1 - auc) * 200), 1)
        else:
            raise ValueError(f"Unknown metric type {x['metric_type']}")

    merged["normalized_value"] = merged.apply(_normalize, axis=1)
    merged_auc_jsd["normalized_value"] = merged_auc_jsd.apply(
        _normalize, axis=1
    )

    # Remove spearman correlation and rename pearson correlation
    merged = merged[~merged["metric_type"].str.contains("spearman-corr")]
    merged["metric_type"] = merged["metric_type"].map(
        lambda x: "norm" if x == "pearson-corr" else x
    )
    merged.to_csv(output_csv, index=False)

    merged["overall_variant"] = merged["variant"] + "-" + merged["experiment"]
    merged_auc_jsd["overall_variant"] = (
        merged_auc_jsd["variant"] + "-" + merged_auc_jsd["experiment"]
    )

    # Plot the results
    plot = (
        ggplot(
            merged, aes(x="metric_type", y="normalized_value", fill="data_type")
        )
        + geom_bar(stat="identity", position="dodge")
        + facet_wrap("~overall_variant", scales="free", ncol=4)
        + labs(x="Metric", y="Quality Score", fill="Data type")
        + ylim(0, 100)
        + theme_bw()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1),
            strip_text_x=element_text(size=8),
            legend_position="top",
        )
    )
    nrow = len(merged["overall_variant"].unique()) // 4
    height = 8 * nrow

    plot.save(output_plot, width=29, height=height, units="cm", limitsize=False)

    plot = (
        ggplot(
            merged_auc_jsd,
            aes(x="metric_type", y="normalized_value", fill="data_type"),
        )
        + geom_boxplot(
            position="dodge",
            outlier_alpha=0.1,
            outlier_shape=".",
            outlier_size=1,
        )
        + facet_wrap("~overall_variant", scales="free", ncol=4)
        + labs(x="Metric", y="Quality Score", fill="Data type")
        + ylim(0, 100)
        + theme_bw()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1),
            strip_text_x=element_text(size=8),
            legend_position="top",
        )
    )
    nrow = len(merged_auc_jsd["overall_variant"].unique()) // 4
    height = 8 * nrow
    plot.save(
        output_plot.with_name(output_plot.stem + "_boxplot.png"),
        width=29,
        height=height,
        units="cm",
        limitsize=False,
    )


if __name__ == "__main__":
    app()
