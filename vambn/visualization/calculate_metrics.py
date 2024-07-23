import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import typer
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_boxplot,
    geom_jitter,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient2,
    theme,
    theme_bw,
    ylim,
)

from vambn.metrics.auc import get_auc
from vambn.metrics.jensen_shannon import jensen_shannon_distance
from vambn.metrics.relative_correlation import RelativeCorrelation
from vambn.utils.logging import setup_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


def drop_irrelevant_columns(
    df: pd.DataFrame, cols: List[str] = ["SUBJID", "VISIT"]
) -> pd.DataFrame:
    """
    Drop irrelevant columns from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols (List[str], optional): List of columns to drop. Defaults to ["SUBJID", "VISIT"].

    Returns:
        pd.DataFrame: The DataFrame with specified columns dropped.
    """
    cols = [x for x in cols if x in df.columns]
    return df.drop(columns=cols)


@app.command()
def calculate_corr_error(
    grouping: Path,
    original_data: Path,
    decoded_data: Path,
    virtual_data: Path,
    all_heatmap_virtual: Path,
    all_heatmap_decoded: Path,
    cont_heatmap_virtual: Path,
    cont_heatmap_decoded: Path,
    result_file: Path,
    dataset_name: str,
    experiment: str,
) -> None:
    """
    Calculate the correlation error and generate heatmaps.

    Args:
        grouping (Path): Path to the grouping CSV file.
        original_data (Path): Path to the original data CSV file.
        decoded_data (Path): Path to the decoded data CSV file.
        virtual_data (Path): Path to the virtual data CSV file.
        all_heatmap_virtual (Path): Path to save the heatmap for all virtual data.
        all_heatmap_decoded (Path): Path to save the heatmap for all decoded data.
        cont_heatmap_virtual (Path): Path to save the heatmap for continuous virtual data.
        cont_heatmap_decoded (Path): Path to save the heatmap for continuous decoded data.
        result_file (Path): Path to save the results JSON file.
        dataset_name (str): The name of the dataset.
        experiment (str): The name of the experiment.
    """
    # Create necessary directories
    all_heatmap_virtual.parent.mkdir(exist_ok=True, parents=True)
    all_heatmap_decoded.parent.mkdir(exist_ok=True, parents=True)
    cont_heatmap_virtual.parent.mkdir(exist_ok=True, parents=True)
    cont_heatmap_decoded.parent.mkdir(exist_ok=True, parents=True)

    # Read and process grouping data
    groups = pd.read_csv(grouping)
    subset = groups.loc[
        ~groups["technical_group_name"].str.match("stalone_"), :
    ]
    continous_cols = set(
        subset.loc[
            subset["hivae_types"].isin(
                ["pos", "real", "truncate_norm", "count", "gamma"]
            ),
            "column_names",
        ].tolist()
    )
    module_cols = set(subset["column_names"].to_list())

    # Read data and check overlap with column sets
    initial = pd.read_csv(original_data)
    virtual = pd.read_csv(virtual_data)
    decoded = pd.read_csv(decoded_data)

    initial_cols = set(initial.columns.to_list())
    decoded_cols = set(decoded.columns.to_list())
    general_overlap = decoded_cols.intersection(initial_cols)
    all_vars_subset = list(general_overlap.intersection(module_cols))
    continous_subset = list(set(all_vars_subset).intersection(continous_cols))

    # Remove rows with NaN values
    initial_nan = initial.loc[:, all_vars_subset].isna().any(axis=1)
    subset_initial = initial.loc[~initial_nan, all_vars_subset]
    subset_virtual = virtual.loc[:, all_vars_subset].dropna()
    subset_decoded = decoded.loc[:, all_vars_subset].dropna()

    assert subset_initial.shape[1] == subset_virtual.shape[1]
    assert subset_initial.shape[1] == subset_decoded.shape[1]

    # SPEARMAN Correlation Error Calculation
    (
        spearman_corr_error_virtual,
        corr_initial,
        corr_virtual,
    ) = RelativeCorrelation.error(subset_initial, subset_virtual)

    assert (
        corr_initial.shape[1] == len(all_vars_subset)
    ), f"corr_initial.shape[1] = {corr_initial.shape[1]}, hivae_subset.__len__() = {len(all_vars_subset)}"
    assert (
        corr_virtual.shape[1] == len(all_vars_subset)
    ), f"corr_virtual.shape[1] = {corr_virtual.shape[1]}, hivae_subset.__len__() = {len(all_vars_subset)}"

    m_init = corr_initial.reset_index().melt("index")
    m_init["type"] = "Real"

    m_virtual = corr_virtual.reset_index().melt("index")
    m_virtual["type"] = "Virtual"

    merged = pd.concat([m_init, m_virtual])
    merged["value"] = merged["value"].astype(float)
    category_order = ["Real", "Virtual"]
    merged["type"] = pd.Categorical(
        merged["type"], categories=category_order, ordered=True
    )
    g = (
        ggplot(
            data=merged,
            mapping=aes(x="index", y="variable", fill="value"),
        )
        + geom_tile()
        + facet_wrap("type")
        + labs(
            title=f"Relative correlation error: {spearman_corr_error_virtual}",
            x="",
            y="",
        )
        + scale_fill_gradient2(
            low="darkblue", mid="lightgrey", high="darkred", midpoint=0
        )
        + theme_bw()
        + theme(axis_text=element_blank(), axis_ticks=element_blank())
    )
    g.save(
        str(all_heatmap_virtual), dpi=300, width=21.7, height=21.7, units="cm"
    )

    (
        spearman_corr_error_decoded,
        corr_initial,
        corr_decoded,
    ) = RelativeCorrelation.error(subset_initial, subset_decoded)

    m_init = corr_initial.reset_index().melt("index")
    m_init["type"] = "Real"

    m_decoded = corr_decoded.reset_index().melt("index")
    m_decoded["type"] = "Decoded"

    merged = pd.concat([m_init, m_decoded])
    merged["value"] = merged["value"].astype(float)
    category_order = ["Real", "Decoded"]
    merged["type"] = pd.Categorical(
        merged["type"], categories=category_order, ordered=True
    )

    g = (
        ggplot(
            data=merged,
            mapping=aes(x="index", y="variable", fill="value"),
        )
        + geom_tile()
        + facet_wrap("type")
        + labs(
            title=f"Relative correlation error: {spearman_corr_error_decoded}",
            x="",
            y="",
        )
        + scale_fill_gradient2(
            low="darkblue", mid="lightgrey", high="darkred", midpoint=0
        )
        + theme_bw()
        + theme(axis_text=element_blank(), axis_ticks=element_blank())
    )
    g.save(
        str(all_heatmap_decoded), dpi=300, width=21.7, height=21.7, units="cm"
    )

    # PEARSON Correlation Error Calculation for Continuous Data
    continous_initial = subset_initial.loc[:, continous_subset]
    continous_virtual = subset_virtual.loc[:, continous_subset]
    continous_decoded = subset_decoded.loc[:, continous_subset]

    (
        pearson_corr_error_virtual,
        corr_initial,
        corr_virtual,
    ) = RelativeCorrelation.error(
        continous_initial, continous_virtual, method="pearson"
    )

    assert (
        corr_initial.shape[1] == len(continous_subset)
    ), f"corr_initial.shape[1] = {corr_initial.shape[1]}, hivae_subset.__len__() = {len(continous_subset)}"
    assert (
        corr_virtual.shape[1] == len(continous_subset)
    ), f"corr_virtual.shape[1] = {corr_virtual.shape[1]}, hivae_subset.__len__() = {len(continous_subset)}"

    m_init = corr_initial.reset_index().melt("index")
    m_init["type"] = "Real"

    m_virtual = corr_virtual.reset_index().melt("index")
    m_virtual["type"] = "Virtual"

    merged = pd.concat([m_init, m_virtual])
    merged["value"] = merged["value"].astype(float)
    category_order = ["Real", "Virtual"]
    merged["type"] = pd.Categorical(
        merged["type"], categories=category_order, ordered=True
    )
    g = (
        ggplot(
            data=merged,
            mapping=aes(x="index", y="variable", fill="value"),
        )
        + geom_tile()
        + facet_wrap("type")
        + labs(
            title=f"Relative correlation error: {pearson_corr_error_virtual}",
            x="",
            y="",
        )
        + scale_fill_gradient2(
            low="darkblue", mid="lightgrey", high="darkred", midpoint=0
        )
        + theme_bw()
        + theme(axis_text=element_blank(), axis_ticks=element_blank())
    )
    g.save(
        str(cont_heatmap_virtual), dpi=300, width=21.7, height=21.7, units="cm"
    )

    (
        pearson_corr_error_decoded,
        corr_initial,
        corr_decoded,
    ) = RelativeCorrelation.error(continous_initial, continous_decoded)

    m_init = corr_initial.reset_index().melt("index")
    m_init["type"] = "Real"

    m_decoded = corr_decoded.reset_index().melt("index")
    m_decoded["type"] = "Decoded"

    merged = pd.concat([m_init, m_decoded])
    merged["value"] = merged["value"].astype(float)
    category_order = ["Real", "Decoded"]
    merged["type"] = pd.Categorical(
        merged["type"], categories=category_order, ordered=True
    )

    g = (
        ggplot(
            data=merged,
            mapping=aes(x="index", y="variable", fill="value"),
        )
        + geom_tile()
        + facet_wrap("type")
        + labs(
            title=f"Relative correlation error: {pearson_corr_error_decoded}",
            x="",
            y="",
        )
        + scale_fill_gradient2(
            low="darkblue", mid="lightgrey", high="darkred", midpoint=0
        )
        + theme_bw()
        + theme(axis_text=element_blank(), axis_ticks=element_blank())
    )
    g.save(
        str(cont_heatmap_decoded), dpi=300, width=21.7, height=21.7, units="cm"
    )

    numeric_results = {
        "spearman_relcorr_virtual": spearman_corr_error_virtual,
        "spearman_relcorr_decoded": spearman_corr_error_decoded,
        "pearson_relcorr_virtual": pearson_corr_error_virtual,
        "pearson_relcorr_decoded": pearson_corr_error_decoded,
        "dataset": dataset_name,
        "experiment": experiment,
    }
    result_file.parent.mkdir(exist_ok=True, parents=True)
    with result_file.open("w+") as f:
        f.write(json.dumps(numeric_results))


@app.command()
def generate_jsd_plot(
    grouping: Path, original_data: Path, comparison_file: Path, jsd_plot: Path
) -> None:
    """
    Generate Jensen-Shannon Distance plot.

    Args:
        grouping (Path): Path to the grouping CSV file.
        original_data (Path): Path to the original data CSV file.
        comparison_file (Path): Path to the comparison data CSV file.
        jsd_plot (Path): Path to save the Jensen-Shannon Distance plot.
    """
    jsd_plot.parent.mkdir(exist_ok=True, parents=True)

    grouping_df = pd.read_csv(grouping)
    module_cols = grouping_df.loc[
        ~grouping_df["technical_group_name"].str.match("stalone_"),
        "column_names",
    ]
    module_cols = set(module_cols.to_list())

    original_df = pd.read_csv(original_data)
    compared_df = pd.read_csv(comparison_file)

    original_cols = set(original_df.columns.to_list())
    compared_cols = set(compared_df.columns.to_list())
    general_overlap = compared_cols.intersection(original_cols)
    module_subset = list(general_overlap.intersection(module_cols))

    subset_original = original_df.loc[:, module_subset]
    subset_compared = compared_df.loc[:, module_subset]

    df_list = []
    for col in module_subset:
        orig_series = subset_original.loc[:, col]
        if orig_series.isna().any():
            logger.info(
                f"Warning: {col} contains NaN values (n = {orig_series.isna().sum()})"
            )
            orig_series.dropna(inplace=True)
        vec_original = orig_series.to_numpy()
        compared_series = subset_compared.loc[:, col]
        if compared_series.isna().any():
            logger.info(
                f"Warning: {col} contains NaN values (n = {compared_series.isna().sum()})"
            )
            compared_series.dropna(inplace=True)
        vec_compared = compared_series.to_numpy()

        dtype = grouping_df.loc[
            grouping_df["column_names"] == col, "hivae_types"
        ].tolist()[0]
        if dtype == "categorical":
            dtype = "cat"
        module = grouping_df.loc[
            grouping_df["column_names"] == col, "technical_group_name"
        ].tolist()[0]
        jsd = jensen_shannon_distance(vec_original, vec_compared, dtype)

        df_list.append(
            {"col": col, "type": dtype, "jsd": jsd, "module": module}
        )

    df = pd.DataFrame(df_list)
    g = (
        ggplot(data=df, mapping=aes(x="module", y="jsd"))
        + geom_boxplot()
        + geom_jitter(alpha=0.6, size=1, color="black")
        + labs(
            title="Distribution of JSDs",
            x="Module",
            y="Jensen-Shannon Distance",
        )
        + ylim(0, 1)
        + theme_bw()
        + theme(axis_text_x=element_text(angle=45, hjust=1, vjust=1))
    )
    g.save(jsd_plot)


@app.command()
def calculate_auc(
    grouping: Path,
    original_data: Path,
    decoded_file: Path,
    virtual_file: Path,
    auc_file: Path,
) -> None:
    """
    Calculate the Area Under the Curve (AUC) for the given datasets.

    Args:
        grouping (Path): Path to the grouping CSV file.
        original_data (Path): Path to the original data CSV file.
        decoded_file (Path): Path to the decoded data CSV file.
        virtual_file (Path): Path to the virtual data CSV file.
        auc_file (Path): Path to save the AUC results CSV file.
    """
    setup_logging(level=logging.INFO)

    grouping_df = pd.read_csv(grouping)
    module_cols = grouping_df.loc[
        ~grouping_df["technical_group_name"].str.match("stalone_"),
        "column_names",
    ]
    module_cols = set(module_cols.to_list())
    modules = set(grouping_df["technical_group_name"].to_list())
    aucs = []
    original_df = pd.read_csv(original_data)
    decoded_df = pd.read_csv(decoded_file)
    virtual_df = pd.read_csv(virtual_file)

    # Relevant columns
    relevant_columns = list(
        set(grouping_df["column_names"].tolist())
        .intersection(original_df.columns.tolist())
        .intersection(decoded_df.columns.tolist())
        .intersection(virtual_df.columns.tolist())
    ) + ["SUBJID", "VISIT"]

    # Order by visit
    original_df = original_df.sort_values("VISIT").loc[:, relevant_columns]
    decoded_df = decoded_df.sort_values("VISIT").loc[:, relevant_columns]
    virtual_df = virtual_df.sort_values("VISIT").loc[:, relevant_columns]

    # Sort and reindex
    original_df = original_df.reindex(sorted(original_df.columns), axis=1)
    decoded_df = decoded_df.reindex(sorted(decoded_df.columns), axis=1)
    virtual_df = virtual_df.reindex(sorted(virtual_df.columns), axis=1)

    assert original_df.shape[1] == decoded_df.shape[1]
    assert original_df.shape[1] == virtual_df.shape[1]

    original_base = original_df.loc[
        original_df["VISIT"] == 1, relevant_columns
    ].drop(columns=["VISIT", "SUBJID"])
    decoded_base = decoded_df.loc[
        decoded_df["VISIT"] == 1, relevant_columns
    ].drop(columns=["VISIT", "SUBJID"])
    virtual_base = virtual_df.loc[
        virtual_df["VISIT"] == 1, relevant_columns
    ].drop(columns=["VISIT", "SUBJID"])

    logger.info("Calculate AUC for all modules and baseline")
    logger.info("Calculate AUC for real vs decoded")
    pauc_decoded, auc_decoded, n_dec = get_auc(
        original_base, decoded_base, n_folds=5
    )
    logger.info("Calculate AUC for real vs virtual")
    pauc_virtual, auc_virtual, n_vir = get_auc(
        original_base, virtual_base, n_folds=5
    )
    logger.info("Calculate AUC for decoded vs virtual")
    pauc_virVdec, auc_virVdec, n_virVdec = get_auc(
        decoded_base, virtual_base, n_folds=5
    )
    aucs.append(
        {
            "module": "all-modules-baseline",
            "pauc_virtual": pauc_decoded,
            "pauc_decoded": pauc_virtual,
            "pauc_virVdec": pauc_virVdec,
            "auc_decoded": auc_decoded,
            "auc_virtual": auc_virtual,
            "auc_virVdec": auc_virVdec,
            "n_virtual": n_vir,
            "n_decoded": n_dec,
            "n_virVdec": n_virVdec,
        }
    )

    # Calculate per module AUC
    for module in modules:
        if "stalone" in module:
            continue
        logger.info(f"Calculating AUC for {module}")

        module_subset = grouping_df.loc[
            grouping_df["technical_group_name"] == module, "column_names"
        ]
        module_subset = set(module_subset.to_list())

        common_cols = module_subset.intersection(original_df.columns)
        common_cols = list(common_cols)
        if len(common_cols) == 0:
            continue

        subset_original = original_df.loc[:, common_cols + ["VISIT"]].dropna()
        max_visit = subset_original["VISIT"].max()
        subset_decoded = decoded_df.loc[
            decoded_df["VISIT"] <= max_visit, common_cols + ["VISIT"]
        ].dropna()
        subset_virtual = virtual_df.loc[
            virtual_df["VISIT"] <= max_visit, common_cols + ["VISIT"]
        ].dropna()
        assert (
            subset_original["VISIT"].unique().tolist()
            == subset_decoded["VISIT"].unique().tolist()
        )
        assert (
            subset_original["VISIT"].unique().tolist()
            == subset_virtual["VISIT"].unique().tolist()
        )

        for col in ["SUBJID", "VISIT"]:
            if col in subset_decoded.columns:
                subset_decoded.drop(col, axis=1, inplace=True)
            if col in subset_virtual.columns:
                subset_virtual.drop(col, axis=1, inplace=True)
            if col in subset_original.columns:
                subset_original.drop(col, axis=1, inplace=True)

        assert subset_original.isna().sum().sum() == 0
        assert subset_decoded.isna().sum().sum() == 0
        assert subset_virtual.isna().sum().sum() == 0
        assert (
            subset_original.columns.to_list()
            == subset_decoded.columns.to_list()
        )

        logger.info("Calculate AUC for real vs decoded")
        pauc_decoded, auc_decoded, n_dec = get_auc(
            subset_original, subset_decoded, n_folds=5
        )

        logger.info("Calculate AUC for real vs virtual")
        pauc_virtual, auc_virtual, n_vir = get_auc(
            subset_original, subset_virtual, n_folds=5
        )

        logger.info("Calculate AUC for decoded vs virtual")
        pauc_virVdec, auc_virVdec, n_virVdec = get_auc(
            subset_decoded, subset_virtual, n_folds=5
        )
        aucs.append(
            {
                "module": module,
                "pauc_virtual": auc_decoded,
                "pauc_decoded": auc_virtual,
                "pauc_virVdec": auc_virVdec,
                "n_virtual": n_vir,
                "n_decoded": n_dec,
                "n_virVdec": n_virVdec,
            }
        )
        logger.info(
            f"AUC for {module} - Decoded: {auc_decoded}, Virtual: {auc_virtual}, Decoded vs Virtual: {auc_virVdec}"
        )

    df = pd.DataFrame(aucs)
    auc_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(auc_file, index=False)

    # Calculate AUC for each module and visit individually
    specific_aucs = []
    for module in modules:
        if "stalone" in module:
            continue
        logger.info(f"Calculating AUC for {module}")
        module_subset = grouping_df.loc[
            grouping_df["technical_group_name"] == module, "column_names"
        ]
        module_subset = set(module_subset.to_list())

        common_cols = module_subset.intersection(original_df.columns)
        common_cols = list(common_cols)
        if len(common_cols) == 0:
            continue

        subset_original = original_df.loc[:, common_cols + ["VISIT"]].dropna()
        max_visit = subset_original["VISIT"].max()
        subset_decoded = decoded_df.loc[
            decoded_df["VISIT"] <= max_visit, common_cols + ["VISIT"]
        ].dropna()
        subset_virtual = virtual_df.loc[
            virtual_df["VISIT"] <= max_visit, common_cols + ["VISIT"]
        ].dropna()
        assert (
            subset_original["VISIT"].unique().tolist()
            == subset_decoded["VISIT"].unique().tolist()
        )
        assert (
            subset_original["VISIT"].unique().tolist()
            == subset_virtual["VISIT"].unique().tolist()
        )

        assert subset_original.isna().sum().sum() == 0
        assert subset_decoded.isna().sum().sum() == 0
        assert subset_virtual.isna().sum().sum() == 0
        assert (
            subset_original.columns.to_list()
            == subset_decoded.columns.to_list()
        )
        for visit in subset_original["VISIT"].unique():
            logger.info(f"Calculating AUC for {module} - Visit {visit}")
            subset_original_visit = subset_original.loc[
                subset_original["VISIT"] == visit
            ]
            subset_decoded_visit = subset_decoded.loc[
                subset_decoded["VISIT"] == visit
            ]
            subset_virtual_visit = subset_virtual.loc[
                subset_virtual["VISIT"] == visit
            ]

            subset_original_visit = drop_irrelevant_columns(
                subset_original_visit
            )
            subset_decoded_visit = drop_irrelevant_columns(subset_decoded_visit)
            subset_virtual_visit = drop_irrelevant_columns(subset_virtual_visit)

            pauc_decoded, auc_decoded, n_dec = get_auc(
                subset_original_visit, subset_decoded_visit, n_folds=5
            )
            pauc_virtual, auc_virtual, n_vir = get_auc(
                subset_original_visit, subset_virtual_visit, n_folds=5
            )
            pauc_virVdec, auc_virVdec, n_virVdec = get_auc(
                subset_decoded_visit, subset_virtual_visit, n_folds=5
            )

            specific_aucs.append(
                {
                    "module": module,
                    "visit": visit,
                    "pauc_virtual": pauc_decoded,
                    "pauc_decoded": pauc_virtual,
                    "pauc_virVdec": pauc_virVdec,
                    "n_virtual": n_vir,
                    "n_decoded": n_dec,
                    "n_virVdec": n_virVdec,
                    "auc_decoded": auc_decoded,
                    "auc_virtual": auc_virtual,
                    "auc_virVdec": auc_virVdec,
                }
            )

    df = pd.DataFrame(specific_aucs)
    specific_auc_path = auc_file.parent / "specific_auc.csv"
    df.to_csv(specific_auc_path, index=False)


if __name__ == "__main__":
    app()
