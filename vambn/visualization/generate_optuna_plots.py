import os
import optuna
import typer
import optuna.visualization as V
from pathlib import Path
import logging

app = typer.Typer()

logger = logging.getLogger(__name__)


@app.command()
def plot_study_results(study_uri: str, study_name: str, output_folder: Path):
    """
    Load the Optuna study from the SQLite database and save various graphics about the conducted study.

    Args:
        study_uri (str): The URI of the SQLite database where the study is stored.
        study_name (str): The name of the study to load.
        output_folder (Path): The directory where the plots will be saved.

    Raises:
        RuntimeError: If no trials are found in the study or if the variance of trials is zero.
    """
    # Load study
    study = optuna.load_study(study_name=study_name, storage=study_uri)

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Check if it's a single or multi-objective study
    is_multi_obj = len(study.directions) > 1

    # Plot param importances
    try:
        fig = V.plot_param_importances(study)
        fig.write_image(os.path.join(output_folder, "param_importances.png"))

        if not is_multi_obj:
            # Plot optimization history
            fig = V.plot_optimization_history(study)
            fig.write_image(
                os.path.join(output_folder, "optimization_history.png")
            )

            # Plot parallel coordinate
            fig = V.plot_parallel_coordinate(study)
            fig.write_image(
                os.path.join(output_folder, "parallel_coordinate.png")
            )

            # Plot slice
            fig = V.plot_slice(study)
            fig.write_image(os.path.join(output_folder, "slice.png"))
        else:
            fig = V.plot_pareto_front(study)
            fig.write_image(os.path.join(output_folder, "pareto_front.png"))

            # Generate the other plots per target
            for i in range(len(study.directions)):

                def target(trial):
                    return trial.values[i]

                target_name = (
                    study.metric_names[i]
                    if study.metric_names
                    else f"Objective {i}"
                )

                # Plot optimization history
                fig = V.plot_optimization_history(
                    study, target=target, target_name=target_name
                )
                fig.write_image(
                    os.path.join(output_folder, f"optimization_history_{i}.png")
                )

                # Plot parallel coordinate
                fig = V.plot_parallel_coordinate(
                    study, target=target, target_name=target_name
                )
                fig.write_image(
                    os.path.join(output_folder, f"parallel_coordinate_{i}.png")
                )

                # Plot slice
                fig = V.plot_slice(
                    study, target=target, target_name=target_name
                )
                fig.write_image(os.path.join(output_folder, f"slice_{i}.png"))

        typer.echo(f"Plots saved in {output_folder}.")
    except RuntimeError:
        logger.error("No trials found in study or variance equals 0.")


if __name__ == "__main__":
    app()
