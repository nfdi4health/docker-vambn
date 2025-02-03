################################################################################
# imports
################################################################################
import json
import logging
import pickle
import re
from functools import reduce
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
import typer

from vambn.data.helpers import prepare_data
from vambn.utils.logging import setup_logging

################################################################################
# global
################################################################################

logger = logging.getLogger()
app = typer.Typer()

app = typer.Typer(help="VAMBN Modelling")
preprocess_app = typer.Typer(help="Preprocess data")
gather_app = typer.Typer(help="Gather data")
app.add_typer(preprocess_app, name="preprocessing")
app.add_typer(gather_app, name="gather")

################################################################################
# main part
################################################################################


@preprocess_app.command()
def make(
    data_file: Path,
    grouping_file: Path,
    groups_file: Path,
    config_json: Path,
    output_path: Path,
    missingness_threshold: int = 50,
    variance_threshold: float = 0.1,
    log_file: Optional[Path] = None,
    scaling: bool = True,
):
    """
    Process and prepare data for VAMBN analysis.

    The function performs the following steps:
        1. Set up logging.
        2. Load configuration settings.
        3. Ensure output directories exist.
        4. Read and preprocess input data.
        5. Filter data based on missingness and variance thresholds.
        6. Prepare data for VAMBN analysis and save it.

    Args:
        data_file (Path): Path to the data file.
        grouping_file (Path): Path to the grouping file.
        groups_file (Path): Path to the file containing module groups.
        config_json (Path): Path to the configuration JSON file.
        output_path (Path): Path to save the processed data.
        missingness_threshold (int, optional): Threshold for missingness. Defaults to 50.
        variance_threshold (float, optional): Minimum variance threshold. Defaults to 0.1.
        log_file (Optional[Path], optional): Path to the log file. Defaults to None.
        scaling (bool, optional): Whether to apply scaling. Defaults to True.
    """
    # set up logging
    setup_logging(level=10, log_file=log_file)

    with config_json.open("r") as f:
        config = json.load(f)

    if "missingness_threshold" in config:
        missingness_threshold = config["missingness_threshold"]
        logger.info(f"Missingness threshold set to {missingness_threshold}%.")
    if "variance_threshold" in config:
        variance_threshold = config["variance_threshold"]
        logger.info(f"Variance threshold set to {variance_threshold}.")

    # ensure output folders exist
    output_path.mkdir(parents=True, exist_ok=True)

    # read in the data files
    data = pd.read_csv(data_file)
    if "SUBJID" not in data.columns:
        data.reset_index(inplace=True)
        if "SUBJID" not in data.columns:
            raise ValueError("SUBJID column not found in data.")
    # drop duplicates based on SUBJID and VISIT
    data.drop_duplicates(subset=["SUBJID", "VISIT"], inplace=True)
    grouping = pd.read_csv(grouping_file)
    with groups_file.open("r") as f:
        selected_modules = f.read().splitlines()
    if len(selected_modules) == 0:
        selected_modules = None

    # ensure that column names contain only two underscores
    def to_camel_case(string_list):
        camel_case_list = []
        for s in string_list:
            parts = s.split("_")
            camel_case = parts[0] + "".join(
                word.capitalize() for word in parts[1:]
            )
            camel_case_list.append(camel_case)
        return camel_case_list

    if any(["_" in x for x in data.columns]):
        data.columns = to_camel_case(data.columns.tolist())
        grouping["column_names"] = to_camel_case(
            grouping["column_names"].tolist()
        )

        # copy old files and save new
        data_file.rename(data_file.parent / f"{data_file.stem}.csv.backup")
        data.to_csv(data_file, index=False)
        grouping_file.rename(
            grouping_file.parent / f"{grouping_file.stem}.csv.backup"
        )
        grouping.to_csv(grouping_file, index=False)

    overall_max_visit = 0
    max_visit_dict = {}

    columns_to_drop = set()
    for data_module in grouping["technical_group_name"].unique():
        columns = tuple(
            set(
                grouping.loc[
                    grouping["technical_group_name"] == data_module,
                    "column_names",
                ].tolist()
                + ["SUBJID", "VISIT"]
            )
        )
        columns = tuple(x for x in columns if x in data.columns)
        if len(columns) == 0:
            logger.warning(
                f"No columns found for module {data_module}. Skipping..."
            )
            continue

        subset = data.loc[:, columns]
        number_of_subjects = subset["SUBJID"].nunique()
        # get the availability ratios of subjects per visit per column
        missingness_ratio = {
            col: 1
            - (subset.groupby("VISIT")[col].count().values / number_of_subjects)
            for col in columns
        }

        module_max_visit = None
        for column, missing_ratios in missingness_ratio.items():
            missing_at_first = missing_ratios[0]
            if missing_at_first > (missingness_threshold / 100):
                logger.info(
                    f"Few data available for column {column}. Dropped due to availability of {missing_at_first} on first visit."
                )
                columns_to_drop.add(column)
                continue
            flag_vector = missing_ratios <= (missingness_threshold / 100)
            max_visit_sum = flag_vector.sum()
            max_visit_validation = flag_vector[: max_visit_sum + 1].sum()
            if max_visit_sum != max_visit_validation:
                raise Exception(
                    f"Missingness is unexpected: {missing_ratios} @ {column}"
                )
            if module_max_visit is None or max_visit_sum < module_max_visit:
                logger.info(
                    f"Module {data_module} has {max_visit_sum} visits with more than {missingness_threshold}% data."
                )
                module_max_visit = max_visit_sum

        if module_max_visit is not None:
            max_visit_dict[data_module] = module_max_visit
            if module_max_visit > overall_max_visit:
                overall_max_visit = module_max_visit
    logger.info(f"Maximum visit to keep: {overall_max_visit}.")
    # filter out visits with higher number
    data = data[data["VISIT"] <= overall_max_visit]

    processed_data = prepare_data(
        data=data,
        grouping=grouping,
        output_path=output_path,
        missingness_threshold=missingness_threshold,
        selected_modules=selected_modules,
        module_wise_features=None,
        max_visit_dict=max_visit_dict,
        scaling=scaling,
        variance_threshold=variance_threshold,
    )

    # plot_dir = output_path / "plots"
    # # generate plots per column
    # for column in tqdm(processed_data.columns.drop("SUBJID")):
    #     # make barplot for categorical data
    #     # make boxplot for numerical data

    #     # get the type of the column
    #     column_type = grouping.loc[
    #         grouping["column_names"] == column, "type"
    #     ].values[0]
    #     if column_type == "categorical" or column_type == "cat":
    #         # make barplot
    #         sns.countplot(data=processed_data, x=column)
    #         plt.savefig(plot_dir / f"{column}_barplot.png")
    #         plt.close()
    #     else:
    #         # make boxplot
    #         sns.boxplot(data=processed_data, x=column)
    #         plt.savefig(plot_dir / f"{column}_boxplot.png")
    #         plt.close()
    # save descriptive statistics
    stats = processed_data.describe().T.sort_values("std", ascending=True)
    # add normalized standard deviation
    stats["std_norm"] = stats["std"] / stats["mean"]
    # round to 2 decimal places
    stats = stats.round(2)
    stats.to_csv(output_path / "descriptive_statistics.csv")

    # save a scatterplot with x=std and y=std_norm
    sns.scatterplot(data=stats, x="std", y="std_norm")
    # make x axis in log scale
    plt.xscale("log")
    plt.savefig(output_path / "std_vs_std_norm.png")

    logger.info("Finished preprocessing data.")


@preprocess_app.command()
def check_specification(
    data_file: Path,
    grouping_file: Path,
    groups_file: Path,
    blacklist_file: Path,
    whitelist_file: Path,
    start_dag_file: Path,
    indicator_file: Path,
    log_file: Optional[Path] = None,
):
    setup_logging(level=10, log_file=log_file)

    data = pl.read_csv(data_file, infer_schema_length=10000)
    grouping = pl.read_csv(grouping_file, infer_schema_length=10000)
    with groups_file.open("r") as f:
        groups = f.read().splitlines()
    blacklist = pl.read_csv(blacklist_file, infer_schema_length=10000)
    whitelist = pl.read_csv(whitelist_file, infer_schema_length=10000)
    start_dag = pl.read_csv(start_dag_file, infer_schema_length=10000)

    logger.info("Data files read successfully.")

    # Check if all column_names from the grouping file are present in the data file
    assert all(
        [col in data.columns for col in grouping["column_names"].to_list()]
    ), "Not all columns from the grouping file are present in the data file"
    logger.info(
        "All columns from the grouping file are present in the data file."
    )

    # Check if all names in the groups file are present in the grouping file (technical_group_name)
    assert all(
        [
            group in grouping["technical_group_name"].to_list()
            for group in groups
        ]
    ), "Not all groups in the groups file are present in the grouping file"
    logger.info(
        "All groups in the groups file are present in the grouping file."
    )

    # Check if the whitelist and blacklist names are valid:
    # Fetch column names of modules starting with stalone
    stalone_cols = (
        grouping.filter(
            grouping["technical_group_name"].str.contains("stalone")
        )
        .select("column_names")["column_names"]
        .to_list()
    )

    # Fetch module names of other modules
    module_names = (
        grouping.filter(
            ~grouping["technical_group_name"].str.contains("stalone")
        )
        .select("technical_group_name")["technical_group_name"]
        .to_list()
    )

    possible_names = module_names + stalone_cols

    def check_list(df, possible_options):
        existing_options = df["from"].to_list() + df["to"].to_list()
        return all([option in possible_options for option in existing_options])

    assert check_list(
        blacklist, possible_names
    ), "Not all blacklist names are valid"
    logger.info("All blacklist names are valid.")
    assert check_list(
        whitelist, possible_names
    ), "Not all whitelist names are valid"
    logger.info("All whitelist names are valid.")
    assert check_list(
        start_dag, possible_names
    ), "Not all start_dag names are valid"
    logger.info("All start_dag names are valid.")

    # Save the indicator file
    indicator_file.parent.mkdir(parents=True, exist_ok=True)
    with open(indicator_file, "w") as f:
        f.write("")
    logger.info("Indicator file saved successfully.")


def extract_module_characteristics(name: str) -> Tuple[str, str]:
    """
    Extract the module name and visit from a given file name.

    Args:
        name (str): File name.

    Returns:
        Tuple[str, str]: Module name and visit.

    Raises:
        ValueError: If the module name or visit cannot be extracted from the file name.
    """
    module_search = re.search("(^[a-zA-Z_0-9]+)_VIS", name)
    if module_search is None:
        raise ValueError(f"Module name could not be extracted from {name}")
    module_name = module_search.group(1)

    visit_search = re.search("_VIS([0-9a-zA-Z]+)_", name)
    if visit_search is None:
        raise ValueError(f"Visit could not be extracted from {name}")
    visit = visit_search.group(1)

    return module_name, visit


def merge_csv_files(
    folder: Path, files: List[Path], suffix: str
) -> pd.DataFrame:
    """
    Read preprocessed CSV files and merge them by a given column.

    Args:
        folder (Path): Folder where the files are located.
        files (List[Path]): Paths to the files.
        suffix (str): Suffix of the files to merge (e.g., '_imp.csv').

    Raises:
        ValueError: If no files are provided or if the number of processed files
            does not match the number of provided files.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    if files is None or len(files) == 0:
        raise ValueError("No files were provided.")

    # use regex to grep module names of the pattern /(^[a-zA-Z_]+)_VIS/
    avail_modules = set()
    avail_visits = set()
    for file in files:
        module, visit = extract_module_characteristics(file.name)
        avail_modules.add(module)
        avail_visits.add(visit)

    logger.info(f"Available modules: {avail_modules}")
    logger.info(f"Available visits: {avail_visits}")

    provided_files = set([str(x) for x in files])
    processed_files = set()
    module_data = []
    for module in avail_modules:
        logger.info(f"Processing module {module}")
        internal_df = []
        for visit in avail_visits:
            logger.info(f"Processing visit {visit}")
            file = folder / f"{module}_VIS{visit}{suffix}"
            if not file.exists():
                logger.warning(f"File {file} does not exist. Skipping...")
                continue
            processed_files.add(str(file))
            visit_data = pd.read_csv(file)
            visit_data["VISIT"] = visit
            # remove _VIS suffix from column names with regex (_VIS[0-9a-zA-Z]+)
            visit_data.columns = [
                re.sub("(_VIS[0-9a-zA-Z]+)", "", x) for x in visit_data.columns
            ]
            if "Unnamed: 0" in visit_data.columns:
                visit_data.drop(columns=["Unnamed: 0"], inplace=True)
            internal_df.append(visit_data)

        # concat along the rows
        internal_df = pd.concat(internal_df, axis=0)
        module_data.append(internal_df)

    # merge the different module data
    overall_data = module_data.pop()
    count = 1
    while module_data:
        count += 1
        overall_data = pd.merge(
            overall_data,
            module_data.pop(),
            how="outer",
            on=["SUBJID", "VISIT"],
        )

    overall_data.insert(0, "VISIT", overall_data.pop("VISIT"))
    overall_data.set_index("SUBJID", inplace=True)

    # check if all files were processed
    if provided_files != processed_files:
        raise ValueError(
            f"Files {provided_files - processed_files} were not processed. Please check the input."
        )

    return overall_data


@preprocess_app.command()
def merge_imputed_data(
    input_folder: Path,
    merged_data: Path,
    transformed_data_path: Path,
    log_file: Optional[Path] = None,
    log_level: int = 20,
) -> None:
    """
    Merge imputed data into a single CSV file.

    Args:
        input_folder (Path): Folder where the files are located.
        merged_data (Path): File where the merged data should be stored.
        transformed_data_path (Path): Path to save the transformed data.
        log_file (Optional[Path], optional): Optional file for logging. Defaults to None.
        log_level (int, optional): Logging level. Defaults to 20.
    """
    setup_logging(level=log_level, log_file=log_file)
    input_files = input_folder.glob("**/*_imp.csv")
    overall_data = merge_csv_files(input_folder, list(input_files), "_imp.csv")
    overall_data.to_csv(str(merged_data))

    transformed_data = overall_data.copy()

    for scaler_file in input_folder.glob("**/*scaler.pkl"):
        # print(f"File: {scaler_file}")
        scaler = pickle.loads(scaler_file.read_bytes())

        column_name = "_".join(
            scaler_file.name.replace("_scaler.pkl", "").split("_")[1:]
        )
        try:
            transformed_data[column_name] = scaler.inverse_transform(
                transformed_data[column_name].values.reshape(-1, 1)
            )
        except KeyError:
            logger.warning(
                f"Could not find column {column_name} in dataframe. Skipping..."
            )
            continue

    transformed_data.to_csv(str(transformed_data_path))


@preprocess_app.command()
def merge_stalone_data(
    input_folder: Path,
    output_file: Path,
    log_file: Optional[Path] = None,
    log_level: int = 20,
) -> None:
    """
    Merge imputed data into a single CSV file.

    Args:
        input_folder (Path): Folder where the files are located.
        output_file (Path): File where the merged data should be stored.
        log_file (Optional[Path], optional): Optional file for logging. Defaults to None.
        log_level (int, optional): Logging level. Defaults to 20.
    """

    setup_logging(level=log_level, log_file=log_file)

    input_files = input_folder.glob("**/stalone*_imp.csv")
    overall_data = merge_csv_files(input_folder, list(input_files), "_imp.csv")
    overall_data.to_csv(output_file)


@preprocess_app.command()
def merge_raw_data(
    input_folder: Path,
    output_file: Path,
    convert: bool = typer.Option(
        False, help="Convert column types if necessary."
    ),
    log_file: Optional[Path] = None,
    log_level: int = 20,
) -> None:
    """
    Merge raw data into a single CSV file.

    Args:
        input_folder (Path): Folder where the files are located.
        output_file (Path): File where the merged data should be stored.
        log_file (Optional[Path], optional): Optional file for logging. Defaults to None.
        log_level (int, optional): Logging level. Defaults to 20.
    """

    setup_logging(level=log_level, log_file=log_file)

    input_files = input_folder.glob("**/*_raw.csv")
    overall_data = merge_csv_files(input_folder, list(input_files), "_raw.csv")
    # sort columns from overall data alphabetically
    overall_data = overall_data.reindex(sorted(overall_data.columns), axis=1)

    if convert:
        encoders = {}
        for encoder_path in input_folder.glob("**/*_label_encoder.pkl"):
            with open(encoder_path, "rb") as f:
                encoder = pickle.load(f)
                prefix = encoder_path.name.replace("_label_encoder.pkl", "")
                column = "_".join(prefix.split("_")[1:])
                encoders[column] = encoder
                if column not in overall_data.columns:
                    raise ValueError(
                        f"Column {column} not found in overall_data."
                    )
                overall_data[column] = encoder.transform(
                    overall_data[column].astype(str)
                )

    overall_data.to_csv(output_file)


def read_and_merge(files: List[Path]) -> pd.DataFrame:
    """
    Read all files and merge them on the columns SUBJID and VISIT.

    Args:
        files (List[Path]): List of files to read.

    Returns:
        pd.DataFrame: Merged dataframe.
    """

    data = []
    for file in files:
        tmp = pd.read_csv(str(file))
        data.append(tmp)

    data = reduce(
        lambda x, y: pd.merge(x, y, on=["SUBJID", "VISIT"], how="outer"), data
    )
    return data


@gather_app.command()
def modular(decoded_folder: Path, input_file: Path, output_data: Path):
    """
    Gather data from decoded files and merge them with the stalone data.

    Args:
        decoded_folder (Path): Path to the folder containing the decoded files.
        input_file (Path): Path to the stalone data.
        output_data (Path): Path to the output file.
    """

    output_data.parent.mkdir(exist_ok=True, parents=True)

    stalone_data = pd.read_csv(input_file)
    decoded_data = read_and_merge(list(decoded_folder.glob("**/*_decoded.csv")))
    # sort columns from decoded data alphabetically
    decoded_data = decoded_data.reindex(sorted(decoded_data.columns), axis=1)
    # assert (
    #     stalone_data.shape[0] == decoded_data.shape[0]
    # ), f"Shapes do not match (stalone: {stalone_data.shape[0]}, decoded: {decoded_data.shape[0]}))"
    merged = pd.merge(
        stalone_data, decoded_data, on=["SUBJID", "VISIT"], how="outer"
    )
    merged.to_csv(output_data, index=False)


@gather_app.command()
def traditional(
    decoded_folders: List[Path],
    input_file: Path,
    output_data: Path,
):
    """
    Gather data from decoded files and merge them with the stalone data.

    Args:
        decoded_folders (List[Path]): List of folders containing the decoded files.
        input_file (Path): Path to the stalone data.
        output_data (Path): Path to the output file.
    """

    output_data.parent.mkdir(exist_ok=True, parents=True)

    stalone_data = pd.read_csv(input_file)
    decoded_data = read_and_merge(
        [n for x in decoded_folders for n in x.glob("**/*_decoded.csv")]
    )
    # sort columns from decoded data alphabetically
    decoded_data = decoded_data.reindex(sorted(decoded_data.columns), axis=1)
    # assert (
    #     stalone_data.shape[0] == decoded_data.shape[0]
    # ), f"Shapes do not match (stalone: {stalone_data.shape[0]}, decoded: {decoded_data.shape[0]}))"
    merged = pd.merge(
        stalone_data, decoded_data, on=["SUBJID", "VISIT"], how="outer"
    )

    merged.to_csv(output_data, index=False)


if __name__ == "__main__":
    app()
