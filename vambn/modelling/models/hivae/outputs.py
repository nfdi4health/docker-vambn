import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar

import pandas as pd
import torch
import typeguard
from torch import Tensor

logger = logging.getLogger()


@dataclass
class EncoderOutput:
    """Dataclass for encoder output.

    Attributes:
        samples_s (Tensor): Samples from the s distribution.
        logits_s (Tensor): Logits for the s distribution.
        mean_z (Tensor): Mean of the z distribution.
        scale_z (Tensor): Scale of the z distribution.
        samples_z (Optional[Tensor]): Samples from the z distribution.
        h_representation (Optional[Tensor]): Hidden representation, if any.
    """


    samples_s: Tensor
    logits_s: Tensor
    mean_z: Tensor
    scale_z: Tensor
    samples_z: Optional[Tensor]
    h_representation: Optional[Tensor] = None

    @property
    def decoder_representation(self) -> Tensor:
        """
        Get the decoder representation.

        Returns:
            Tensor: The hidden representation if available, otherwise the samples from the z distribution.
        """
        return (
            self.h_representation
            if self.h_representation is not None
            else self.samples_z
        )

    @decoder_representation.setter
    def decoder_representation(self, value: Tensor) -> None:
        """
        Set the decoder representation.

        Args:
            value (Tensor): The value to set as the hidden representation.
        """
        # if value.shape != self.samples_z.shape:
        #     raise ValueError(
        #         f"Shape of value ({value.shape}) does not match shape of samples_z ({self.samples_z.shape})"
        #     )

        self.h_representation = value


# Output Classes
@dataclass
class DecoderOutput:
    """Dataclass to hold the output from the decoder.

    Attributes:
        log_p_x (Tensor): Log-likelihood of the data.
        kl_z (Tensor): KL divergence for the z variable.
        kl_s (Tensor): KL divergence for the s variable.
        corr_loss (Tensor): Correlation loss if applicable.
        recon (Optional[Tensor]): Reconstruction, if any.
        samples (Optional[Tensor]): Samples generated, if any.
        enc_s (Optional[Tensor]): Encoded s values, if any.
        enc_z (Optional[Tensor]): Encoded z values, if any.
        output_name (Optional[str]): Name of the output, if any.
        detached (bool): Whether the tensors have been detached.
    """


    log_p_x: Tensor
    kl_z: Tensor
    kl_s: Tensor
    corr_loss: Tensor
    recon: Optional[Tensor] = None
    samples: Optional[Tensor] = None
    enc_s: Optional[Tensor] = None
    enc_z: Optional[Tensor] = None
    output_name: Optional[str] = None
    detached: bool = False

    def __post_init__(self):
        """
        Validate dimensions of the log-likelihood tensor.
        
        Raises:
            Exception: If log-likelihood tensor is not of dimension 1.
        """
        if self.log_p_x.ndim != 1:
            raise Exception(
                f"Log-likelihood is not of correct dimension ({self.log_p_x.ndim}, expected 1)"
            )

    def __str__(self) -> str:
        """
        String representation of the DecoderOutput object.

        Returns:
            str: A string describing the DecoderOutput object.
        """

        if self.output_name is not None:
            return f"Decoder output for {self.output_name})"
        else:
            return f"Decoder output (id={id(self)})"

    def detach_and_move(self) -> "DecoderOutput":
        """
        Detach all tensors and move them to CPU.

        Returns:
            DecoderOutput: The detached DecoderOutput object.
        """

        self.detached = True
        self.log_p_x = self.log_p_x.detach().cpu()
        self.kl_z = self.kl_z.detach().cpu()
        self.kl_s = self.kl_s.detach().cpu()
        if self.corr_loss is not None:
            self.corr_loss = self.corr_loss.detach().cpu()
        if self.samples is not None:
            self.samples = self.samples.detach().cpu()
        if self.enc_s is not None:
            self.enc_s = self.enc_s.detach().cpu()
        if self.enc_z is not None:
            self.enc_z = self.enc_z.detach().cpu()
        return self

    @property
    def elbo(self) -> Tensor:
        """
        Calculate the negative Evidence Lower Bound (ELBO).

        Returns:
            Tensor: The negative ELBO.
        """

        return self.log_p_x - self.kl_z - self.kl_s

    @property
    def loss(self) -> Tensor:
        """
        Calculate the loss based on the negative ELBO.

        Returns:
            Tensor: The loss tensor.

        Raises:
            Exception: If tensors have been detached.
        """

        if self.detached:
            logger.error("Cannot calculate loss. Tensors have been detached.")
            raise Exception(
                "Cannot calculate loss. Tensors have been detached."
            )
        loss = -self.elbo.sum()
        return loss


@dataclass
class LogLikelihoodOutput:
    """Dataclass to hold the output from log-likelihood functions.

    Attributes:
        log_p_x (Tensor): Log-likelihood for observed data.
        log_p_x_missing (Tensor): Log-likelihood for missing data.
        samples (Optional[Tensor]): Samples generated, if any.
    """


    log_p_x: Tensor
    log_p_x_missing: Tensor
    samples: Optional[Tensor] = None


@dataclass
class HivaeOutput:
    """Dataclass for HIVAE output.

    Attributes:
        loss (Tensor): Loss tensor.
        enc_z (Tensor): Encoded z values.
        enc_s (Tensor): Encoded s values.
        samples (Optional[Tensor]): Samples generated, if any.
        n (Optional[int]): Number of samples.
        single (bool): Whether this is a single output.
    """

    loss: Tensor
    enc_z: Tensor
    enc_s: Tensor
    samples: Optional[Tensor] = None
    n: Optional[int] = None
    single: bool = True

    def __post_init__(self):
        """
        Initialize the number of samples.
        """
        self.n = self.enc_z.shape[0]

    @property
    def n_loss(self) -> int:
        """
        Get the number of loss values.

        Returns:
            int: Number of loss values.
        """
        return 1 if self.loss.ndim == 0 else self.loss.shape[0]

    def detach(self) -> "HivaeOutput":
        """
        Detach all tensors and move them to CPU.

        Returns:
            HivaeOutput: The detached HivaeOutput object.
        """
        self.loss = self.loss.detach().cpu()
        self.enc_z = self.enc_z.detach().cpu()
        self.enc_s = self.enc_s.detach().cpu()

        if self.samples is not None:
            self.samples = self.samples.detach().cpu()
        return self

    @property
    def avg_loss(self) -> float:
        """
        Calculate the average loss.

        Returns:
            float: The average loss.

        Raises:
            ValueError: If the loss tensor has an invalid dimension.
        """
        if self.loss.ndim == 0:
            return float(self.loss)
        elif self.loss.ndim == 1:
            return float(self.loss.mean())
        else:
            raise ValueError(
                f"Loss is of wrong dimension ({self.loss.ndim}), expected 0 or 1"
            )

    def stack(self, other: "HivaeOutput") -> "HivaeOutput":
        """
        Stack another HivaeOutput object with this one.

        Args:
            other (HivaeOutput): The other HivaeOutput object to stack.

        Returns:
            HivaeOutput: The stacked HivaeOutput object.
        """

        self.single = False
        self.loss = torch.cat([self.loss.view(-1), other.loss.view(-1)])
        self.enc_z = torch.cat([self.enc_z, other.enc_z])
        self.enc_s = torch.cat([self.enc_s, other.enc_s])

        if self.samples is not None:
            self.samples = torch.cat([self.samples, other.samples])
        self.n += other.n
        return self

    def __add__(self, other: "HivaeOutput") -> "HivaeOutput":
        """
        Add another HivaeOutput object to this one.

        Args:
            other (HivaeOutput): The other HivaeOutput object to add.

        Returns:
            HivaeOutput: The resulting HivaeOutput object.
        """

        return self.stack(other)


@dataclass
class LstmHivaeOutput(HivaeOutput):
    """Dataclass for LSTM HIVAE output."""

    pass


HivaeOutputs = HivaeOutput | LstmHivaeOutput


@dataclass
class ModularHivaeOutput:
    """Dataclass for modular HIVAE output.

    Attributes:
        outputs (Tuple[HivaeOutputs, ...]): Tuple of HIVAE outputs. See HivaeOutputs for details.
    """

    outputs: Tuple[HivaeOutputs, ...]

    def __add__(self, other: "ModularHivaeOutput") -> "ModularHivaeOutput":
        """
        Add another ModularHivaeOutput object to this one.

        Args:
            other (ModularHivaeOutput): The other ModularHivaeOutput object to add.

        Returns:
            ModularHivaeOutput: The resulting ModularHivaeOutput object.
        """
        for old, new in zip(self.outputs, other.outputs):
            old += new
        logger.debug(f"Added {other} to {self}")
        return self

    def detach(self) -> "ModularHivaeOutput":
        """
        Detach all tensors in the outputs and move them to CPU.

        Returns:
            ModularHivaeOutput: The detached ModularHivaeOutput object.
        """
        for output in self.outputs:
            output.detach()
        return self

    @property
    def avg_loss(self) -> float:
        """
        Calculate the average loss across all outputs.

        Returns:
            float: The average loss.
        """
        return sum([x.avg_loss for x in self.outputs])

    @property
    def loss(self) -> Tensor:
        """
        Calculate the total loss across all outputs.

        Returns:
            Tensor: The total loss tensor.
        """
        return torch.stack([x.loss for x in self.outputs]).sum()

    def __iter__(self):
        """
        Iterate over the outputs.

        Returns:
            Iterator: An iterator over the outputs.
        """
        return iter(self.outputs)

    def __len__(self):
        """
        Get the number of outputs.

        Returns:
            int: The number of outputs.
        """
        return len(self.outputs)

    def __item__(self, idx: int) -> HivaeOutputs:
        """
        Get an output by index.

        Args:
            idx (int): The index of the output to retrieve.

        Returns:
            HivaeOutputs: The output at the specified index.
        """
        return self.outputs[idx]


@dataclass
class HivaeEncoding:
    """Dataclass for HIVAE encoding.

    Attributes:
        s (torch.Tensor): Encoding for s.
        z (torch.Tensor): Encoding for z.
        module (str): Module name.
        samples (Optional[Tensor]): Samples generated, if any.
        subjid (Optional[List[str | int]]): Subject IDs.
        h_representation (Optional[Tensor]): Hidden representation, if any.
    """

    s: torch.Tensor
    z: torch.Tensor
    module: str
    samples: Optional[Tensor] = None
    subjid: Optional[List[str | int]] = None
    h_representation: Optional[Tensor] = None

    def __post_init__(self):
        """
        Initialize the encoding and ensure tensors are on CPU and have the correct dtype.
        """

        if self.s.device != torch.device("cpu"):
            self.s = self.s.cpu()
        if self.z.device != torch.device("cpu"):
            self.z = self.z.cpu()
        if self.samples is not None and self.samples.device != torch.device(
            "cpu"
        ):
            self.samples = self.samples.cpu()

        if self.s.dtype != torch.float32:
            self.s = self.s.float()

        # make sure z encoding is float
        if self.z.dtype != torch.float32:
            self.z = self.z.float()

    @property
    def decoder_representation(self) -> Tensor:
        """
        Get the decoder representation.

        Returns:
            Tensor: The hidden representation if available, otherwise the z encoding.
        """
        return (
            self.h_representation
            if self.h_representation is not None
            else self.z
        )

    @decoder_representation.setter
    def decoder_representation(self, value: Tensor) -> None:
        """
        Set the decoder representation.

        Args:
            value (Tensor): The value to set as the hidden representation.
        """
        self.h_representation = value

    def convert(self) -> Dict[str, List[str | float]]:
        """
        Convert the encoding to a dictionary format.

        Returns:
            Dict[str, List[str | float]]: The converted encoding.
        """
        data = {
            f"{self.module}_s": self.s.argmax(dim=1).tolist(),
            "SUBJID": self.subjid,
        }
        if self.z.ndim == 2 and self.z.shape[1] > 1:
            for i in range(self.z.shape[1]):
                data[f"{self.module}_z{i}"] = self.z[:, i].tolist()
        else:
            data[f"{self.module}_z"] = self.z.view(-1).tolist()
        return data

    def get_samples(self, module: Optional[str] = None) -> Tensor:
        """
        Get the samples for the specified module.

        Args:
            module (Optional[str]): The module name. If None, return all samples.

        Returns:
            Tensor: The samples tensor.
        """
        if module is None:
            return self.samples
        else:
            assert self.module is not None
            assert module == self.module
            return self.samples

    @typeguard.typechecked
    def save_meta_enc(self, path: Path):
        """
        Save the metadata encoding to a CSV file.

        Args:
            path (Path): The file path to save the metadata.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        converted = self.convert()
        df = pd.DataFrame(converted)
        df.to_csv(path, index=False)


@dataclass
class ModularHivaeEncoding:
    """Dataclass for modular HIVAE encoding.

    Attributes:
        encodings (Tuple[HivaeEncoding, ...]): Tuple of HIVAE encodings. See HivaeEncoding for details.
        modules (List[str]): List of module names in the same order as encodings.
    """
    encodings: Tuple[HivaeEncoding, ...]
    modules: List[str]

    def __post_init__(self):
        """
        Initialize the modular encoding and ensure tensors are on CPU and have the correct dtype.
        
        Raises:
            Exception: If modules in encodings do not match modules in ModularHivaeEncoding.
        """
        for encoding in self.encodings:
            if encoding.s.device != torch.device("cpu"):
                encoding.s = encoding.s.cpu()
            if encoding.z.device != torch.device("cpu"):
                encoding.z = encoding.z.cpu()
            if (
                encoding.samples is not None
                and encoding.samples.device != torch.device("cpu")
            ):
                encoding.samples = encoding.samples.cpu()

            if any(
                [x.module != y for x, y in zip(self.encodings, self.modules)]
            ):
                raise Exception(
                    "Modules in encodings do not match modules in ModularHivaeEncoding"
                )

    def convert(self) -> Dict[str, List[float | str]]:
        """
        Convert the modular encoding to a dictionary format.

        Returns:
            Dict[str, List[float | str]]: The converted encoding.
        """
        out = {}
        for encoding in self.encodings:
            data = encoding.convert()
            out.update(data)
        return out

    @typeguard.typechecked
    def save_meta_enc(self, path: Path):
        """
        Save the metadata encoding to a CSV file.

        Args:
            path (Path): The file path to save the metadata.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        converted = self.convert()
        df = pd.DataFrame(converted)
        df.to_csv(path, index=False)

    def get_samples(
        self, module: Optional[str] = None
    ) -> Tensor | Dict[str, Tensor]:
        """
        Get the samples for the specified module.

        Args:
            module (Optional[str]): The module name. If None, return all samples.

        Returns:
            Tensor | Dict[str, Tensor]: The samples tensor or a dictionary of samples.
        """
        if module is None:
            raise {x.module: x.samples for x in self.encodings}
        else:
            selected_encodings = [
                x for x in self.encodings if x.module == module
            ]
            assert len(selected_encodings) == 1
            return selected_encodings[0].samples

    def __getitem__(self, idx: int) -> HivaeEncoding:
        """
        Get an encoding by index.

        Args:
            idx (int): The index of the encoding to retrieve.

        Returns:
            HivaeEncoding: The encoding at the specified index.
        """
        return self.encodings[idx]

    def get(self, module: str) -> HivaeEncoding:
        """
        Get an encoding by module name.

        Args:
            module (str): The module name.

        Returns:
            HivaeEncoding: The encoding for the specified module.

        Raises:
            Exception: If the module is not found in encodings.
        """
        for encoding in self.encodings:
            if encoding.module == module:
                return encoding
        raise Exception(f"Module {module} not found in encodings")


# Generic type for output
ModelOutputType = TypeVar("ModelOutputType", bound=HivaeOutput)
