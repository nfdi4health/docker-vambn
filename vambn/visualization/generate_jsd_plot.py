from pathlib import Path

import pandas as pd
import typer
from plotnine import (
    aes,
    coord_flip,
    geom_boxplot,
    geom_jitter,
    ggplot,
    theme_bw,
)


def main(results_file: Path, output_file: Path) -> None:
    """
    Generate a boxplot with jitter from a results CSV file and save it as an image.

    Args:
        results_file (Path): Path to the input CSV file containing results.
        output_file (Path): Path to save the output image file.
    """
    assert output_file.parent.exists(), "Output directory does not exist."

    plot_df = pd.read_csv(str(results_file))

    g = (
        ggplot(plot_df, aes("module_name", "jsd", color="column"))
        + geom_boxplot()
        + geom_jitter(alpha=0.6, size=1, color="black")
        + coord_flip()
        + theme_bw()
    )
    g.save(str(output_file), dpi=300)


if __name__ == "__main__":
    typer.run(main)
