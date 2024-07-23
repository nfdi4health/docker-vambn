import optuna
import typer
from optuna.trial import TrialState


def main(db_url: str, study_name: str) -> None:
    """
    Get the number of completed or pruned trials and return to bash.

    Args:
        db_url (str): URL to the Optuna database.
        study_name (str): Name of the study.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=db_url)
        completed_or_pruned_trials = sum(
            trial.state in [TrialState.COMPLETE, TrialState.PRUNED]
            for trial in study.trials
        )

        typer.echo(completed_or_pruned_trials)
    except KeyError:
        # In case the study is not found, output 0
        typer.echo(0)


if __name__ == "__main__":
    typer.run(main)
