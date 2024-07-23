import logging
import pickle
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

from vambn.data.dataclasses import VariableType
from vambn.data.datasets import ModuleDataset, VambnDataset

logger = logging.getLogger(__name__)


def load_vambn_data(
    data_folder: Path,
    selected_visits: Optional[List[int]] = None,
    selected_modules: Optional[List[str]] = None,
) -> VambnDataset:
    """
    Load the data from the preprocessed folder.

    Args:
        data_folder (Path): Folder containing the preprocessed data.
        selected_visits (Optional[List[int]], optional): List of visits to select. Defaults to None.
        selected_modules (Optional[List[str]], optional): List of modules to select. Defaults to None.

    Raises:
        FileNotFoundError: If the data folder or any required data file is not found.

    Returns:
        VambnDataset: Dataset with the loaded data.
    """
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder {data_folder} not found")
    data_files = [x for x in data_folder.glob("**/*imp.csv")]
    module_names = [str(x).split("/")[-1].split("_imp")[0] for x in data_files]

    modules_map = defaultdict(lambda: defaultdict())

    for i, module_name in enumerate(module_names):
        if "stalone" in module_name:
            continue
        # Read data
        logger.debug(f"Processing module {module_name}")
        file = str(data_folder / f"{module_name}_imp.csv")
        module_data = pd.read_csv(file, index_col=0)
        data = torch.from_numpy(module_data.values)
        # Read column (features) names
        column_names = module_data.columns.values
        # Read row (sample) names
        row_names = module_data.index.values

        scaler_files = list(
            data_folder.glob(f"**/{module_name.split('_')[0]}*_scaler.pkl")
        )
        scalers = {}
        for scaler_file in scaler_files:
            scaler = pickle.loads(scaler_file.open("rb").read())
            column_name = scaler_file.stem.replace(
                f"{module_name.split('_')[0]}_", ""
            ).replace("_scaler", "")
            column_name_vis = f"{column_name}_{module_name.split('_')[1]}"
            scalers[column_name_vis] = scaler

        # Read types of columns
        type_df = pd.read_csv(str(data_folder / f"{module_name}_types.csv"))

        def get_type(type_row: pd.Series) -> VariableType:
            return VariableType(
                name=type_row["column_names"],
                data_type=type_row["type"],
                n_parameters=int(
                    1
                    if type_row["type"] == "count"
                    else 2
                    if type_row["type"]
                    in ["real", "pos", "truncate_norm", "gamma"]
                    else type_row["nclass"]
                    if type_row["type"] == "cat"
                    else 0
                ),
                input_dim=int(
                    type_row["nclass"] if type_row["type"] == "cat" else 1,
                ),
                scaler=scalers.get(type_row["column_names"])
                if type_row["type"] in ["real", "pos", "truncate_norm", "gamma"]
                else None,
            )

        types = [get_type(row) for _, row in type_df.iterrows()]

        # convert scalers into the correct list
        scaler_list = [None] * len(column_names)
        for i, name in enumerate(column_names):
            if name in scalers:
                scaler_list[i] = scalers[name]

        # assert that data matches type
        for i, (var_type, column) in enumerate(zip(types, column_names)):
            data_column = data[:, i].nan_to_num()
            if var_type.data_type == "count":
                assert torch.all(
                    data_column >= 0
                ), f"Negative values in column {column}"

            logger.info(
                f"Range of values for column {column} in module {module_name} ({var_type.data_type}): {data_column.min()} -- {data_column.max()}"
            )

        # Missing data
        # if 1 = observed, 0 = missing
        # TODO: save mask directly instead of having the longtable
        missing_mask = torch.ones(size=data.shape)
        try:
            missing = pd.read_csv(str(data_folder / f"{module_name}_mask.csv"))
            # assert column names "row" and "column"
            assert "row" in missing.columns.values
            assert "column" in missing.columns.values

            for tup in missing.itertuples():
                missing_mask[tup.row - 1, tup.column - 1] = 0
        except pd.errors.EmptyDataError:
            pass
        except FileNotFoundError:
            logger.warning(
                "File with missing observations not found. Assume that all values are present"
            )

        # Verify that torch.isnan(data) == missing_mask
        assert torch.all(
            torch.isnan(data) == (missing_mask == 0)
        ), "Missing mask does not match data"

        # Create mapping of module id to data
        modules_map[module_name]["data"] = data
        modules_map[module_name]["column_names"] = column_names
        modules_map[module_name]["row_names"] = row_names
        modules_map[module_name]["types"] = types
        modules_map[module_name]["missing_mask"] = missing_mask
        modules_map[module_name]["scalers"] = tuple(scaler_list)

    logger.info(f"Loaded {len(modules_map)} modules.")
    logger.info(f"Module names: {sorted(list(modules_map.keys()))}")

    module_data = []
    for module_name, data_dictionary in modules_map.items():
        if "VIS" in module_name:
            name = "_".join(module_name.split("_")[:-1])
            visit = re.sub("[A-Z]+", "", module_name.split("_")[-1])
        else:
            name = module_name
            visit = "1"
        types = data_dictionary["types"]

        if len(types) >= 1:
            module_data.append(
                ModuleDataset(
                    name=name,
                    data=pd.DataFrame(data_dictionary["data"].numpy()),
                    mask=pd.DataFrame(data_dictionary["missing_mask"].numpy()),
                    variable_types=types,
                    visit_number=int(visit),
                    scalers=data_dictionary["scalers"],
                    columns=data_dictionary["column_names"],
                    subjects=data_dictionary["row_names"],
                )
            )
        else:
            logger.warning(f"Module {module_name} has no variable. Skipping.")

    logger.info(f"Loaded {len(module_data)} modules.")
    ds = VambnDataset(module_data)
    if selected_visits is not None or selected_modules is not None:
        if selected_visits is not None and not isinstance(
            selected_visits, list
        ):
            selected_visits = [selected_visits]

        if selected_modules is not None and not isinstance(
            selected_modules, list
        ):
            selected_modules = [selected_modules]

        if selected_visits is not None and len(selected_visits) == 0:
            selected_visits = None

        if selected_modules is not None and len(selected_modules) == 0:
            selected_modules = None

        ds.select_modules(
            visits=selected_visits,
            selection=selected_modules,
        )

    # print selected visits and modules from ds
    logger.info(f"Selected visits: {ds.selected_visits}")
    logger.info(f"Selected modules: {ds.selected_modules}")

    return ds


def filter_data(
    data: pd.DataFrame,
    missingness_threshold: float,
    selected_columns: List[str] | None,
    variance_threshold: float = 0.1,
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Filter data by removing columns with zero variance and too many missing values.

    Args:
        data (pd.DataFrame): Input data.
        missingness_threshold (float): Threshold for missingness.
        selected_columns (List[str] | None): Columns to keep.
        variance_threshold (float, optional): Minimum variance. Defaults to 0.1.

    Returns:
        Tuple[pd.DataFrame, Set[str]]: Dataframe with filtered columns and set of selected columns.
    """
    original_data = data.copy()

    if selected_columns is None:
        subj_columns = set([x for x in data.columns if x.startswith("SUBJID")])
        data.drop(columns=list(subj_columns), inplace=True)

        # Filter columns that have only unique values
        selected_unique_columns = set()
        for column in data.columns:
            if data[column].nunique() > 1:
                selected_unique_columns.add(column)

        print(
            f"Ratio of selected unique columns: {len(selected_unique_columns) / data.shape[1]}"
        )

        # Function to compute 'variance' for numeric and 'diversity' for string columns
        def compute_variance_or_diversity(column):
            if column.dtype in [np.int64, np.float64]:
                # return column.var(skipna=True)
                x = column.copy()
                # compute the variance
                return x.var(skipna=True)
            elif column.dtype == "object":  # If column has string data
                unique_count = column.nunique()
                return 10000 if unique_count > 1 else 0
            return 10000 if column.nunique() > 1 else 0

        column_measures = data.apply(compute_variance_or_diversity)
        print(column_measures)
        selected_var_columns = column_measures[
            column_measures > variance_threshold
        ].keys()
        logger.info(
            f"Ratio of selected var columns: {len(selected_var_columns) / data.shape[1]}"
        )

        column_missingness = data.isna().sum(axis=0) / data.shape[0]
        selected_missingness_columns = column_missingness[
            column_missingness < (missingness_threshold / 100)
        ].keys()
        logger.info(
            f"Ratio of selected missingness columns: {len(selected_missingness_columns) / data.shape[1]}"
        )

        # Get the overlap of both column sets
        selected_columns = (
            set(selected_var_columns)
            & set(selected_missingness_columns)
            & set(selected_unique_columns)
        )
        selected_columns |= set(
            data.columns[data.dtypes == "object"]
        )  # Include string columns
        selected_columns |= subj_columns  # Add SUBJID columns back
        logger.info(
            f"Ratio of selected columns: {len(selected_columns) / data.shape[1]}"
        )

    # Sort the selected columns to maintain a consistent order
    selected_columns = sorted(list(selected_columns))

    # Filter the original data to include only the selected columns
    filtered_data = original_data.loc[:, selected_columns].copy()

    return filtered_data, set(selected_columns)


def prepare_data(
    data: pd.DataFrame,
    grouping: pd.DataFrame,
    output_path: Path,
    missingness_threshold: float,
    max_visit_dict: Dict[str, int],
    selected_modules: List[str] | None,
    module_wise_features: Dict[str, Optional[Set[str]]] | None,
    selected_visits: List[int] | None = None,
    scaling: bool = True,
    variance_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Prepare data for VAMBN and save it in the respective output folder.

    The function performs the following steps:
        1. Iterate over the modules and timepoints/visits.
        2. Filter out columns with zero variance and too many missing values.
        3. Keep track of missing values and create a mask (1 = missing, 0 = not missing).
        4. Impute missing data for standalone variables.
        5. Save imputed and raw data, as well as types and missing mask for each module.

    Args:
        data (pd.DataFrame): Input data.
        grouping (pd.DataFrame): DataFrame containing grouping information.
        output_path (Path): Path to save the processed data.
        missingness_threshold (float): Threshold for missingness.
        max_visit_dict (Dict[str, int]): Dictionary with maximum visit number for each module.
        selected_modules (List[str] | None): List of modules to select.
        module_wise_features (Dict[str, Optional[Set[str]]] | None): Features for each module.
        selected_visits (List[int] | None, optional): List of visits to select. Defaults to None.
        scaling (bool, optional): Whether to apply scaling. Defaults to True.
        variance_threshold (float, optional): Minimum variance threshold. Defaults to 0.1.

    Returns:
        pd.DataFrame: Prepared data.
    """
    if selected_visits is not None:
        data = data.copy().loc[data["VISIT"].isin(selected_visits), :]

    if data.shape[0] == 0:
        raise Exception(
            f"No data left after filtering. Check selected visits. ({selected_visits})"
        )

    # available_visits = sorted(data["VISIT"].unique().tolist())
    available_subjects = sorted(data["SUBJID"].unique().tolist())
    # sorted(grouping["technical_group_name"].unique().tolist())
    grouping.sort_values(
        by=["technical_group_name", "column_names"], inplace=True
    )
    grouping.drop_duplicates(inplace=True)

    subject_df = pd.DataFrame(available_subjects, columns=["SUBJID"])

    if module_wise_features is None:
        module_wise_features: Dict[str, Optional[Set[str]]] = dict()

    overall_data = []

    for module_name, subset in grouping.groupby("technical_group_name"):
        assert isinstance(module_name, str)
        # Skip if module is not selected
        if (
            "stalone" not in module_name
            and selected_modules is not None
            and module_name not in selected_modules
        ):
            logger.info(f"Skipping group {module_name}.")
            continue
        else:
            logger.info(f"Processing group {module_name}.")

        # get the features for this module
        possible_features = subset["column_names"].tolist() + [
            "VISIT",
            "SUBJID",
        ]
        selected_features = list(
            set([x for x in possible_features if x in data.columns])
        )

        # get the data for this module
        module_data = data.loc[:, selected_features].copy()
        selected_columns = module_wise_features.get(module_name, None)

        if module_data["VISIT"].min() != 1:
            raise Exception("Minimum visit is not 1.")

        final_module_data = []

        max_vist = int(max_visit_dict.get(module_name, 0))
        for visit in range(1, max_vist + 1):
            if selected_columns is not None:
                # rename all columns in selected_columns with current visit
                # use regex to replace the part after _VIS with the current visit
                selected_columns = [
                    re.sub("_VIS[a-bA-B0-9]+", f"_VIS{visit}", x)
                    for x in selected_columns
                ]

            logger.info(f"Processing visit {visit}.")
            logger.debug(f"Selected columns: {selected_columns}.")
            # get the visit data
            visit_data = module_data[module_data["VISIT"] == visit].copy()

            # drop the visit column
            visit_data.drop(columns=["VISIT"], inplace=True)

            # merge with subject_df to ensure all subjects are present
            visit_data = subject_df.merge(visit_data, on="SUBJID", how="left")

            # rename all columns by appending VIS_{visit}
            visit_data.rename(
                columns={
                    x: f"{x}_VIS{visit}" if x not in ["SUBJID", "VISIT"] else x
                    for x in visit_data.columns
                },
                inplace=True,
            )
            if "stalone" in module_name:
                visit_data.rename(
                    columns={
                        x: f"SA_{x}" if x not in ["SUBJID", "VISIT"] else x
                        for x in visit_data.columns
                    },
                    inplace=True,
                )

            # filter data
            visit_data, selected_columns = filter_data(
                data=visit_data,
                missingness_threshold=missingness_threshold,
                selected_columns=selected_columns,
                variance_threshold=variance_threshold,
            )
            if len(selected_columns) <= 1:
                logger.warning(
                    f"No columns left after filtering in module {module_name} at visit {visit}."
                )
                raise Exception("No columns left after filtering.")
            else:
                logger.info(f"Selected columns: {selected_columns}.")

            if "SUBJID" in visit_data.columns:
                visit_data.set_index("SUBJID", inplace=True)

            # get the missing mask; output should be a dataframe with 1 = missing, 0 = not missing of the same shape as visit_data
            missing_mask = visit_data.isna().astype(int)

            # define auxiliary data
            # if "stalone" in module name this is equal to missing mask
            # else check row-wise if all values are missing
            if "stalone" in module_name:
                auxiliary_data = missing_mask.copy()
                auxiliary_data.rename(
                    columns={
                        x: f"AUX_{x}" if x not in ["SUBJID", "VISIT"] else x
                        for x in auxiliary_data.columns
                    },
                    inplace=True,
                )
            else:
                auxiliary_data = missing_mask.all(axis=1).astype(int)
                auxiliary_data.columns = [f"AUX_{module_name}_VIS{visit}"]

            # impute missing values if standalone variable
            # if the variable type is continuous (real, pos), use mean imputation
            # else use mode imputation

            # adapt the name of the subset to the current visit
            renamed_subset = subset.copy()
            renamed_subset["column_names"] = renamed_subset[
                "column_names"
            ].apply(
                lambda x: f"{x}_VIS{visit}"
                if x not in ["SUBJID", "VISIT"]
                else x
            )
            if "stalone" in module_name:
                renamed_subset["column_names"] = renamed_subset[
                    "column_names"
                ].apply(
                    lambda x: f"SA_{x}" if x not in ["SUBJID", "VISIT"] else x
                )

            # create subset filtered and keep the order of selected columns
            subset_filtered = renamed_subset.loc[
                renamed_subset["column_names"].isin(selected_columns), :
            ].copy()
            subset_filtered.sort_values(by=["column_names"], inplace=True)

            # make sure the order of "column_names" is identical to the visit_data.columns
            if "stalone" in module_name:
                # iterate over the columns and types
                for column, type in zip(
                    subset_filtered["column_names"],
                    subset_filtered["hivae_types"],
                ):
                    if type in ["real", "pos", "truncate_norm", "gamma"]:
                        mean_val = float(visit_data[column].mean())
                        visit_data[column].fillna(mean_val, inplace=True)
                        logger.info(f"Imputed {column} with mean {mean_val}.")
                    else:
                        option = visit_data[column].mode(dropna=True).tolist()
                        if len(option) > 1:
                            logger.warning(
                                f"Multiple modes found for {column}. Using the first one."
                            )
                            option = option[0]
                        elif len(option) == 0:
                            logger.warning(
                                f"No mode found for {column}. Imputing with 0."
                            )
                            raise Exception("No mode found.")
                        elif len(option) == 1:
                            option = option[0]

                        # convert to float if numeric
                        # if isinstance(option, (int, float)):
                        #     option = float(option)
                        visit_data[column].fillna(option, inplace=True)
                        logger.info(f"Imputed {column} with mode {option}.")

                # check if there are still missing values
                if visit_data.isna().sum().sum() > 0:
                    logger.warning(
                        f"Module {module_name} still contains missing values after imputation."
                    )
                    raise Exception("Missing values after imputation.")

            # convert the missingness mask to a long format with the row and column indices of the missing values
            missing_mask_long = pd.melt(
                missing_mask.reset_index(),
                id_vars=["SUBJID"],
                var_name="column",
                value_name="missing",
            )
            missing_mask_long["row"] = missing_mask_long["SUBJID"].apply(
                lambda x: visit_data.index.get_loc(x) + 1
            )
            missing_mask_long["column"] = missing_mask_long["column"].apply(
                lambda x: visit_data.columns.get_loc(x) + 1
            )
            missing_mask_long = missing_mask_long.loc[
                :, ["row", "column", "missing"]
            ]
            missing_mask_long = missing_mask_long[
                missing_mask_long["missing"] == 1
            ]
            if missing_mask_long.shape[0] == 0:
                missing_mask_long = pd.DataFrame({"row": [], "column": []})
            else:
                missing_mask_long.drop(columns=["missing"], inplace=True)

            # finally create a types dataframe with the columns (type, dim, nclass)
            # type => hivae_types of the respective column
            # dim => 1 if continous (real, pos, count), number_of_classes if categorical
            # nclass => number_of_classes if categorical, 0 if continuous
            subset_filtered["dim"] = None
            subset_filtered["nclass"] = None
            for column, type in zip(
                subset_filtered["column_names"], subset_filtered["hivae_types"]
            ):
                if type in ["real", "pos", "truncate_norm", "count", "gamma"]:
                    subset_filtered.loc[
                        subset_filtered["column_names"] == column, "dim"
                    ] = 1
                    subset_filtered.loc[
                        subset_filtered["column_names"] == column, "nclass"
                    ] = ""
                else:
                    # determine the number of classes
                    nclass = visit_data[column].nunique()
                    subset_filtered.loc[
                        subset_filtered["column_names"] == column, "dim"
                    ] = nclass
                    subset_filtered.loc[
                        subset_filtered["column_names"] == column, "nclass"
                    ] = nclass

            types = subset_filtered[
                [
                    "hivae_types",
                    "dim",
                    "nclass",
                    "column_names",
                ]
            ].copy()
            # rename categorical to cat in hivae_types
            types["hivae_types"] = types["hivae_types"].apply(
                lambda x: "cat" if x == "categorical" else x
            )
            types.rename({"hivae_types": "type"}, axis=1, inplace=True)

            # get raw data with same features and order
            raw_data = module_data[module_data["VISIT"] == visit].copy()
            # rename all columns by appending VIS_{visit}
            raw_data.rename(
                columns={
                    x: f"{x}_VIS{visit}" if x not in ["SUBJID", "VISIT"] else x
                    for x in raw_data.columns
                },
                inplace=True,
            )

            # add "SA_" for stalone variables
            if "stalone" in module_name:
                raw_data.rename(
                    columns={
                        x: f"SA_{x}" if x not in ["SUBJID", "VISIT"] else x
                        for x in raw_data.columns
                    },
                    inplace=True,
                )
            raw_data_merged = subject_df.merge(
                raw_data, on="SUBJID", how="left"
            )
            raw_data_filtered = raw_data_merged.loc[
                :, list(selected_columns)
            ].copy()
            raw_data_filtered.set_index("SUBJID", inplace=True)

            if visit_data.columns.tolist() != types["column_names"].tolist():
                raise Exception(
                    f"Column names do not match. Missing columns: {set(visit_data.columns) - set(types['column_names'])} or {set(types['column_names']) - set(visit_data.columns)}"
                )

            # print info about missingness
            for column in visit_data.columns:
                logger.info(
                    f"Missingness in module {module_name} at visit {visit}: {column} - {visit_data[column].isna().sum() / visit_data.shape[0]}"
                )

            if visit_data.shape[1] == 0:
                logger.warning(
                    f"No columns left after filtering in module {module_name} at visit {visit}."
                )
                raise Exception("No columns left after filtering.")

            logger.info(
                f"Saving data for module {module_name} at visit {visit} with {visit_data.shape[1]} columns."
            )
            # save the data
            visit_data.to_csv(
                output_path / f"{module_name}_VIS{visit}_imp.csv",
            )
            missing_mask_long.to_csv(
                output_path / f"{module_name}_VIS{visit}_mask.csv", index=False
            )
            auxiliary_data.to_csv(
                output_path / f"{module_name}_VIS{visit}_aux.csv",
            )
            types.to_csv(
                output_path / f"{module_name}_VIS{visit}_types.csv", index=False
            )
            raw_data_filtered.to_csv(
                output_path / f"{module_name}_VIS{visit}_raw.csv",
            )
            final_module_data.append(raw_data_filtered)

        # append selected modules to module_wise_features
        if module_name not in module_wise_features:
            module_wise_features[module_name] = selected_columns

        # concat all dataframes for this module vertically/stacked
        filtered_module_df = pd.concat(final_module_data, axis=0)
        overall_data.append(filtered_module_df)

    # merge all dataframes by subject
    overall_data_df = reduce(
        lambda x, y: pd.merge(x, y, on="SUBJID", how="outer"), overall_data
    )

    return overall_data_df
