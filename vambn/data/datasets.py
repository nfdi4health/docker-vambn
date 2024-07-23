import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset

from vambn.data.dataclasses import (
    VariableType,
    check_equal_types,
    get_input_dim,
)
from vambn.data.scalers import LogStdScaler

logger = logging.getLogger(__name__)


class IterDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        missing_mask: torch.Tensor,
        types: List[VariableType],
    ) -> None:
        """
        Initialize the IterDataset.

        Args:
            data (torch.Tensor): Tensor containing the data.
            missing_mask (torch.Tensor): Tensor with the corresponding missing mask (0=missing, 1=available).
            types (List[VariableType]): List of VariableType, each specifying dtype, ndim, and nclasses.

        Raises:
            ValueError: If data or missing_mask is not 2-dimensional, or if data contains NaN values.
        """
        if data.ndim != 2 or missing_mask.ndim != 2:
            raise ValueError(
                "Both data and missing_mask tensors must be 2-dimensional"
            )

        # Vector with Samples x Features
        self.data = data
        self.missing_mask = missing_mask
        self.types = types
        self.num_visits = 1

        if torch.isnan(self.data).any():
            raise ValueError("Data contains NaN values, which are not allowed.")

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get the data and the missing mask for a specific index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Tensor, Tensor]: Data and missing mask for the sample.
        """
        data = self.data[idx, :]
        mask = self.missing_mask[idx, :]

        return data, mask

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def subset(self, idx: List[int]) -> "IterDataset":
        """
        Create a subset of the dataset.

        Args:
            idx (List[int]): Indices of the samples to be selected.

        Returns:
            IterDataset: Subset of the dataset.
        """
        return IterDataset(
            self.data[idx, :], self.missing_mask[idx, :], self.types
        )

    @property
    def ndim(self) -> int:
        """
        Input dimensionality of the dataset.

        Returns:
            int: Dimensionality of the dataset.
        """
        return get_input_dim(self.types)

    def to(self, device: torch.device) -> "IterDataset":
        """
        Move the dataset to a specified device.

        Args:
            device (torch.device): The device to move the dataset to.

        Returns:
            IterDataset: The dataset moved to the specified device.
        """
        self.data = self.data.to(device)
        self.missing_mask = self.missing_mask.to(device)
        return self


class LongitudinalDataset(Dataset):
    """
    Dataset for longitudinal data, where each sample consists of multiple visits/timepoints.

    Attributes:
        data (torch.Tensor): Tensor containing the data.
        missing_mask (torch.Tensor): Tensor indicating missing data (0=missing, 1=available).
        types (List[VariableType]): List of VariableType objects containing dtype, ndim, and nclasses.
    """

    def __init__(
        self,
        data: torch.Tensor,
        missing_mask: torch.Tensor,
        types: List[VariableType],
    ) -> None:
        """
        Initialize the LongitudinalDataset.

        Args:
            data (torch.Tensor): Tensor containing the data.
            missing_mask (torch.Tensor): Tensor indicating missing data (0=missing, 1=available).
            types (List[VariableType]): List of VariableType objects containing dtype, ndim, and nclasses.

        Raises:
            ValueError: If `data` or `missing_mask` is not a 3-dimensional tensor.
            ValueError: If `data` contains NaN values.
        """
        if data.ndim != 3 or missing_mask.ndim != 3:
            raise ValueError(
                "Both data and missing_mask tensors must be 3-dimensional"
            )

        # Array with Time x Samples x Features
        self.data = data
        self.missing_mask = missing_mask
        self.types = types

        if torch.isnan(self.data).any():
            raise ValueError("Data contains NaN values, which are not allowed.")

    def to(self, device: torch.device) -> "LongitudinalDataset":
        """
        Move the dataset to the specified device.

        Args:
            device (torch.device): The device to which the data and mask should be moved.

        Returns:
            LongitudinalDataset: The dataset on the specified device.
        """
        self.data = self.data.to(device)
        self.missing_mask = self.missing_mask.to(device)
        return self

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get the longitudinal data and the missing mask for a specific index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Tensor, Tensor]: 3D tensor with the data and 3D tensor with the missing mask for the sample.
        """
        s_data = self.data[:, idx, :]
        s_mask = self.missing_mask[:, idx, :]

        return s_data, s_mask

    def __len__(self) -> int:
        """
        Number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.data.shape[1]

    def subset(self, idx: List[int]) -> "LongitudinalDataset":
        """
        Create a subset of the dataset.

        Args:
            idx (List[int]): Indices of the samples to be selected.

        Returns:
            LongitudinalDataset: Subset of the dataset.
        """
        return LongitudinalDataset(
            self.data[:, idx, :], self.missing_mask[:, idx, :], self.types
        )

    @property
    def ndim(self) -> int:
        """
        Input dimensionality of the dataset.

        Returns:
            int: Input dimensionality of the dataset.
        """
        return get_input_dim(self.types)

    @property
    def num_visits(self) -> int:
        """
        Number of visits/timepoints in the dataset.

        Returns:
            int: Number of visits/timepoints.
        """
        return self.data.shape[0]


@dataclass
class ModuleDataset:
    """
    A class to represent a module dataset.

    Attributes:
        name (str): The name of the dataset.
        data (pd.DataFrame): The data of the dataset.
        mask (pd.DataFrame): The mask for the dataset.
        variable_types (List[VariableType]): The variable types for the dataset.
        scalers (Tuple[Optional[StandardScaler | LogStdScaler]]): The scalers for the dataset.
        columns (List[str]): The columns of the dataset.
        subjects (List[str]): The subjects in the dataset.
        visit_number (int): The visit number. Defaults to 1.
        id_name (Optional[str]): The ID name for the dataset. Defaults to None.
        ndim (int): The number of dimensions. Defaults to -1.
        device (torch.device): The device to use. Defaults to torch.device("cpu").
        move_to_device (bool): Whether to move the data to the specified device. Defaults to True.
    """

    name: str
    data: pd.DataFrame
    mask: pd.DataFrame
    variable_types: List[VariableType]
    scalers: Tuple[Optional[StandardScaler | LogStdScaler]]

    columns: List[str]
    subjects: List[str]
    visit_number: int = 1
    id_name: Optional[str] = None
    ndim: int = -1
    device: torch.device = torch.device("cpu")
    move_to_device: bool = True

    def __post_init__(self):
        """
        Validate and initialize the dataset attributes after the object is created.

        Raises:
            Exception: If the length of variable_types does not match the number of columns in data.
            Exception: If the length of columns does not match the number of columns in data or variable_types.
            Exception: If the length of subjects does not match the number of rows in data.
        """
        if len(self.variable_types) != self.data.shape[1]:
            raise Exception("Types do not match to the data")

        if self.columns is None:
            logger.warning("No columns found, using column names from data")
            self.columns = self.data.columns.tolist()

        if self.subjects is None or len(self.subjects) != self.data.shape[0]:
            logger.warning("No subjects found, using index as subjects")
            self.subjects = self.data.index.tolist()

        self.ndim = get_input_dim(self.variable_types)
        if self.id_name is None:
            self.id_name = f"{self.name}_VIS{self.visit_number}"

        if len(self.columns) != self.data.shape[1] or len(self.columns) != len(
            self.variable_types
        ):
            raise Exception("Columns do not match to the data")

        if len(self.subjects) != self.data.shape[0]:
            raise Exception(
                f"Subjects do not match to the data. Found shape {self.data.shape[0]} and {len(self.subjects)} subjects."
            )

    def to(self, device: torch.device) -> "ModuleDataset":
        """
        Move the dataset to a specific device.

        Args:
            device (torch.device): Device to be used.

        Returns:
            ModuleDataset: Dataset on the specified device.
        """
        self.device = device
        return self

    @property
    def input_data(self) -> Tensor:
        """
        Get the input data as a tensor with nan values replaced by 0.

        Returns:
            Tensor: Data tensor.
        """
        x = torch.tensor(self.data.values).float()
        if self.move_to_device:
            x = x.to(self.device)
        return x.nan_to_num()

    @property
    def input_mask(self) -> Tensor:
        """
        Get the input mask as a tensor with nan values replaced by 0.

        Returns:
            Tensor: Mask tensor.
        """
        mask = torch.tensor(self.mask.values).float()
        if self.move_to_device:
            mask = mask.to(self.device)
        return mask.nan_to_num()

    def subset(self, idx: List[int] | np.ndarray) -> "ModuleDataset":
        """
        Subset the data and mask by a list of indices.

        Args:
            idx (List[int] | np.ndarray): Indices of the samples to be selected.

        Returns:
            ModuleDataset: New ModuleDataset object with the subsetted data and mask.
        """
        return ModuleDataset(
            name=self.name,
            data=self.data.iloc[idx, :],
            mask=self.mask.iloc[idx, :],
            variable_types=self.variable_types,
            scalers=self.scalers,
            columns=self.columns,
            subjects=self.subjects,
            visit_number=self.visit_number,
            id_name=self.id_name,
            ndim=self.ndim,
        )

    @property
    def pytorch_dataset(self) -> IterDataset:
        """
        Get a PyTorch compatible dataset based on the given data and mask.

        Returns:
            IterDataset: PyTorch compatible dataset.
        """
        return IterDataset(
            self.input_data, self.input_mask, self.variable_types
        )

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the data and mask to pandas DataFrame.

        Returns:
            pd.DataFrame: Data and mask as pandas DataFrame.
        """
        out_df = self.data.copy()
        out_df.columns = [re.sub(r"_VIS\d+", "", x) for x in self.columns]
        out_df["SUBJID"] = self.subjects
        out_df["VISIT"] = self.visit_number
        return out_df

    def __str__(self) -> str:
        """
        String representation of the ModuleDataset object.

        Returns:
            str: A string representation of the ModuleDataset.
        """
        return f"ModuleData ({self.name}, {self.visit_number})"


class VambnDataset(Dataset):
    """
    Dataset for the VAMBN model.

    Attributes:
        modules (List[ModuleDataset]): List of module datasets.
        module_names (List[str]): List of unique module names.
        num_patients (int): Number of patients in the dataset.
        visits_per_module (Optional[dict]): Dictionary of visits per module.
        selected_modules (List[str]): List of selected modules.
        selected_visits (List[int]): List of selected visits.
        num_timepoints (int): Number of timepoints.
        subj (List[str]): List of subject IDs.
        device (torch.device): Device to use for tensor operations.
    """

    def __init__(self, modules: List[ModuleDataset]) -> None:
        """
        Initialize the VambnDataset.

        Args:
            modules (List[ModuleDataset]): Modules to be included in the dataset.

        Raises:
            Exception: If no modules are provided or if the number of rows in the modules do not match.
        """
        super().__init__()

        if len(modules) < 1:
            raise Exception("No modules found")

        self.modules = sorted(modules, key=lambda x: x.name)
        self.module_names = sorted(list(set([x.name for x in modules])))

        unique_nrow = list(set([x.data.shape[0] for x in modules]))
        if len(unique_nrow) > 1:
            raise Exception(f"Number of rows do not match: {unique_nrow}")

        self.num_patients = unique_nrow[0]
        self.visits_per_module = None

        self._prepare_dataset(self.module_names)
        self.selected_modules = self.module_names
        self.selected_visits = list(set(self.module_wise_visits))
        self.num_timepoints = len(list(set(self.module_wise_visits)))
        self.subj = self.modules[0].subjects
        self.device = torch.device("cpu")

        if self.subj is None:
            raise Exception("No subjects found")

    def to(self, device: torch.device) -> "VambnDataset":
        """
        Move the dataset to a specific device.

        Args:
            device (torch.device): Device to be used.

        Returns:
            VambnDataset: Dataset on the specified device.
        """
        self.device = device
        self.modules = [x.to(device) for x in self.modules]
        return self

    def get_modules(self, name: str) -> List[ModuleDataset]:
        """
        Get the modules with the given name.

        Args:
            name (str): Name of the module.

        Returns:
            List[ModuleDataset]: ModuleDataset objects with the given name sorted by visit number.
        """
        selection = [x for x in self.modules if name == x.name]
        selection = sorted(selection, key=lambda x: x.visit_number)
        return selection

    def __getitem__(self, idx: int) -> tuple[list[Tensor], list[Tensor]]:
        """
        Get the data and mask for a specific index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Data and mask tensors.
        """
        s_data = self.patient_x[idx]
        s_mask = self.patient_mask[idx]
        return s_data, s_mask  # , self.subj[idx]

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.num_patients

    def __str__(self) -> str:
        """
        String representation of the VambnDataset object.

        Returns:
            str: A string representation of the VambnDataset.
        """
        return f"""
        VambnDataset
            Modules: {self.module_names}
            N: {self.num_patients}
            V: {self.num_timepoints}
            M: {self.num_modules}
        """

    @property
    def num_modules(self) -> int:
        """
        Get the number of modules in the dataset.

        Returns:
            int: Number of modules.
        """
        return len(self.module_names)

    @property
    def is_longitudinal(self) -> bool:
        """
        Check if the dataset is longitudinal.

        Returns:
            bool: True if the dataset is longitudinal, False otherwise.
        """
        return self.num_timepoints > 1

    def _prepare_dataset(
        self,
        selection: Optional[List[str]] = None,
        visits: Optional[List[int]] = None,
    ):
        """
        Prepare the dataset by preparing the internal variables.

        Args:
            selection (Optional[List[str]]): List of module names to select. Defaults to None.
            visits (Optional[List[int]]): List of visits to select. Defaults to None.

        Raises:
            Exception: If types do not match across visits.
        """
        if selection is not None:
            self.selected_modules = selection
            self.module_names = selection
        if visits is not None and visits != []:
            self.selected_visits = visits
            self.modules = [
                x
                for x in self.modules
                if x.name in self.module_names and x.visit_number in visits
            ]
            self.num_timepoints = len(visits)
        else:
            self.modules = [
                x for x in self.modules if x.name in self.module_names
            ]

        self.module_wise_visits = tuple(
            set([x.visit_number for x in self.modules])
        )
        logger.debug(f"Module wise visits: {self.module_wise_visits}")
        self.num_timepoints = len(self.module_wise_visits)
        logger.debug(f"Number of timepoints: {self.num_timepoints}")
        self.module_wise_names = [x.name for x in self.modules]

        module_data = {}
        module_mask = {}
        for sel in self.module_names:
            modules = self.get_module_data(sel)
            if len(modules) < 1:
                raise Exception(
                    f"No modules found for {sel}, available: {self.module_names}, module_names: {tuple(x.name for x in self.modules)})"
                )
            if len(modules) > 1:
                unequal_types = [
                    check_equal_types(
                        modules[0].variable_types, x.variable_types
                    )
                    for x in modules[1:]
                ]
                if any(unequal_types):
                    logger.info(
                        f"Module types: {[x.variable_types for x in modules]}"
                    )
                    raise Exception(f"Types do not match across visits ({sel})")

            logger.info(f"Module {sel} has {len(modules)} visits")
            x = []
            mask = []
            for module in modules:
                x.append(module.input_data)
                mask.append(module.input_mask)

            n_features = x[0].shape[1]
            n_rows = x[0].shape[0]

            module_data[sel] = (
                torch.stack(x, dim=1).float().view(n_rows, -1, n_features)
                if self.is_longitudinal
                else x[0].float()
            )
            module_mask[sel] = (
                torch.stack(mask, dim=1).float().view(n_rows, -1, n_features)
                if self.is_longitudinal
                else mask[0].float()
            )
            logger.info(f"Module {sel} has shape {module_data[sel].shape}")

        self.x = module_data
        self.mask = module_mask

        self.patient_x = []
        self.patient_mask = []
        for i in range(self.num_patients):
            if self.is_longitudinal:
                x = [self.x[sel][i, :, :] for sel in self.module_names]
                mask = [self.mask[sel][i, :, :] for sel in self.module_names]
            else:
                x = [self.x[sel][i, :] for sel in self.module_names]
                mask = [self.mask[sel][i, :] for sel in self.module_names]
            self.patient_x.append(x)
            self.patient_mask.append(mask)

        self.visits_per_module = {}
        for x in self.modules:
            if x.name not in self.visits_per_module:
                self.visits_per_module[x.name] = 0
            self.visits_per_module[x.name] += 1

    def get_module(self, name: str) -> ModuleDataset:
        """
        Get the ModuleDataset for a given name.

        Args:
            name (str): Name of the module with visit number.

        Returns:
            ModuleDataset: ModuleDataset object with the given name.

        Raises:
            Exception: If the module is not found or multiple modules with the same name are found.
        """
        selection = [x for x in self.modules if name == x.id_name]
        if len(selection) != 1:
            logger.warning(
                f"Selection: {selection}, Modules: {[x.id_name for x in self.modules]}, Name: {name}"
            )
            raise Exception(
                f"Selection: {selection}, Modules: {[x.id_name for x in self.modules]}, Name: {name}"
            )
        return selection[0]

    def get_module_data(self, selection: str) -> List[ModuleDataset]:
        """
        Get the ModuleDataset for a given name without considering visit number.

        Args:
            selection (str): Name of the module.

        Returns:
            List[ModuleDataset]: List of all ModuleDataset objects with the given name sorted by visit number.
        """
        modules = sorted(
            [x for x in self.modules if x.name == selection],
            key=lambda x: x.visit_number,
        )
        return modules

    def get_longitudinal_data(self, selection: str) -> Tuple[Tensor, Tensor]:
        """
        Get longitudinal data for a specific module.

        Args:
            selection (str): Module name.

        Returns:
            Tuple[Tensor, Tensor]: Data and mask tensors.
        """
        modules = self.get_modules(selection)
        data = torch.stack(
            [torch.tensor(x.data.values) for x in modules], dim=1
        )
        mask = torch.stack(
            [torch.tensor(x.mask.values) for x in modules], dim=1
        )
        assert data.ndim == 3

        # if the data is longitudinal, then the shape is (n_subjects, n_visits, n_features)
        # otherwise remove the second dimension
        if not self.is_longitudinal:
            data = data.squeeze(1)
            mask = mask.squeeze(1)

        return data, mask

    def select_modules(
        self,
        selection: Optional[List[str]] = None,
        visits: Optional[List[int]] = None,
    ):
        """
        Select certain modules and visits from the existing dataset.

        Args:
            selection (Optional[List[str]]): Module names. Defaults to None.
            visits (Optional[List[int]]): Visit numbers. Defaults to None.
        """
        if selection is None and visits is None:
            return None
        else:
            if selection is not None and isinstance(selection, str):
                selection = [selection]

            if visits is not None and isinstance(visits, int):
                visits = [visits]

            self._prepare_dataset(selection=selection, visits=visits)

    def subset(self, ratio: float) -> "VambnDataset":
        """
        Subset the dataset by a given ratio.

        Args:
            ratio (float): Ratio of the subset to be returned.

        Returns:
            VambnDataset: Subset of the dataset.
        """
        patient_idx = np.arange(self.num_patients)
        selected_idx = np.random.choice(
            patient_idx, size=round(self.num_patients * ratio)
        )
        return self.subset_by_idx(selected_idx)

    def subset_by_idx(
        self, selected_idx: List[int] | np.ndarray[Any, np.dtype]
    ) -> "VambnDataset":
        """
        Subset the dataset by a given list of indices.

        Args:
            selected_idx (List[int]): Indices of the samples to be selected.

        Returns:
            VambnDataset: Subset of the dataset.
        """
        out_modules = [x.subset(selected_idx) for x in self.modules]
        out_ds = VambnDataset(out_modules)
        return out_ds

    def train_test_split(
        self, test_ratio: float
    ) -> Tuple["VambnDataset", "VambnDataset"]:
        """
        Generate a train and test split of the dataset.

        Args:
            test_ratio (float): Ratio of the dataset to be used as the test set.

        Returns:
            Tuple[VambnDataset, VambnDataset]: Train and test split.
        """
        idx = list(range(self.num_patients))
        train_idx, test_idx = train_test_split(
            idx, test_size=test_ratio, random_state=42
        )
        return self.subset_by_idx(train_idx), self.subset_by_idx(test_idx)

    def get_iter_dataset(self, name: str) -> IterDataset | LongitudinalDataset:
        """
        Get a PyTorch compatible dataset for a given module name.

        Args:
            name (str): Module name.

        Returns:
            IterDataset | LongitudinalDataset: Either an IterDataset or a LongitudinalDataset depending on the number of visits.
        """
        modules = self.get_modules(name)
        if self.is_longitudinal and len(modules) > 1:
            data = torch.stack([x.input_data for x in modules], dim=0)
            mask = torch.stack([x.input_mask for x in modules], dim=0)
            return LongitudinalDataset(data, mask, modules[0].variable_types)
        else:
            assert len(modules) == 1
            data = modules[0].input_data
            mask = modules[0].input_mask
            return IterDataset(data, mask, modules[0].variable_types)

    def to_pandas(self, module_name: Optional[str] = None) -> pd.DataFrame:
        """
        Convert the data and mask to pandas DataFrame.

        Args:
            module_name (Optional[str]): Name of the module to convert. Defaults to None.

        Returns:
            pd.DataFrame: Data and mask as pandas DataFrame.
        """
        if module_name is not None:
            selected_modules = [
                x for x in self.modules if x.name == module_name
            ]
        else:
            selected_modules = self.modules
        module_dfs = {}
        for module in selected_modules:
            if module.name not in module_dfs:
                module_dfs[module.name] = module.to_pandas()
            else:
                module_dfs[module.name] = pd.concat(
                    [module_dfs[module.name], module.to_pandas()]
                )
        # merge on subject id and visit
        df = None
        for m in module_dfs.values():
            if df is None:
                df = m
            else:
                df = df.merge(m, on=["SUBJID", "VISIT"], how="outer")

        return df
