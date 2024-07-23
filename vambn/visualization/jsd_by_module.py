import typer
import pandas as pd
from pathlib import Path
import json
from plotnine import (
    element_blank,
    geom_errorbar,
    ggplot,
    aes,
    geom_bar,
    facet_wrap,
    labs,
    position_dodge,
    theme,
    element_text,
    ylim,
)


def main(jsd_metrics: Path, grouping: Path, output: Path):
    """
    Generate a bar plot with error bars showing the Jensen-Shannon Divergence (JSD)
    for different modules and types, based on the provided JSD metrics and grouping file.

    Args:
        jsd_metrics (Path): Path to the JSON file containing JSD metrics.
        grouping (Path): Path to the CSV file containing column names and their respective types and module mappings.
        output (Path): Path to save the generated plot.
    """
    with jsd_metrics.open("r") as f:
        jsd_metrics_dict = json.load(f)
    grouping_df = pd.read_csv(grouping)

    type_dict = {
        row["column_names"]: row["hivae_types"]
        for _, row in grouping_df.iterrows()
    }
    module_dict = {
        row["column_names"]: row["technical_group_name"]
        for _, row in grouping_df.iterrows()
    }
    jsd_df = pd.DataFrame(jsd_metrics_dict)
    jsd_df["types"] = jsd_df["column"].map(type_dict)
    jsd_df["module"] = jsd_df["column"].map(module_dict)

    agg_metrics = jsd_df.groupby(["module", "types"]).aggregate(
        {
            "jsd_decoded": ["mean", "std"],
            "jsd_virtual": ["mean", "std"],
        }
    )

    # Melt the aggregated DataFrame to long format
    agg_metrics_melted = agg_metrics.reset_index()
    agg_metrics_melted.columns = [
        "_".join(col).strip("_") for col in agg_metrics_melted.columns.values
    ]
    agg_metrics_melted = agg_metrics_melted.melt(
        id_vars=["module", "types"], var_name="metric", value_name="value"
    )

    # Split the "metric" column into separate "variant" and "stat" columns
    metric_stat_df = agg_metrics_melted["metric"].str.split("_", expand=True)
    metric_stat_df.columns = ["metric", "variant", "stat"]
    agg_metrics_melted = pd.concat(
        [
            agg_metrics_melted.loc[:, ["module", "types", "value"]],
            metric_stat_df,
        ],
        axis=1,
    )
    agg_metrics_melted = agg_metrics_melted.drop("metric", axis=1)

    agg_wide = agg_metrics_melted.pivot_table(
        index=["module", "types", "variant"],
        columns="stat",
        values="value",
    ).reset_index()
    agg_wide["lower"] = agg_wide["mean"] - agg_wide["std"]
    agg_wide["upper"] = agg_wide["mean"] + agg_wide["std"]

    # Generate bar plot with error bars and facets per type (virtual/decoded)
    plot = (
        ggplot(
            agg_wide,
            aes(x="module", y="mean", fill="types"),
        )
        + geom_bar(stat="identity", position="dodge")
        + geom_errorbar(
            aes(ymin="lower", ymax="upper"),
            position=position_dodge(width=0.9),
            width=0.2,
        )
        + ylim(0, 1)
        + labs(y="Jensen-Shannon Divergence", title="JSD by module")
        + facet_wrap("~variant", scales="free_x")
        + theme(
            axis_text_x=element_text(angle=45, hjust=1),
            axis_title_x=element_blank(),
            figure_size=(10, 6),
        )
    )

    # Save the plot to the output file
    plot.save(output, dpi=300)


if __name__ == "__main__":
    typer.run(main)
