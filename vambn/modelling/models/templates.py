import logging
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Tuple, TypeVar

import mlflow
import numpy as np
import torch
from lightning import Fabric
from torch import Tensor, nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from vambn.modelling.models.conversion import Conversion

BatchInput = TypeVar("BatchInput")
BaseOutput = TypeVar("BaseOutput")
ForwardOutput = TypeVar("ForwardOutput")
NewBatchInput = TypeVar("NewBatchInput")
NewBaseOutput = TypeVar("NewBaseOutput")
EncodingInput = TypeVar("EncodingInput")
NewForwardOutput = TypeVar("NewForwardOutput")
OptimizerInput = TypeVar("OptimizerInput")
SchedulerInput = TypeVar("SchedulerInput")
LearningRateInput = TypeVar("LearningRateInput")
logger = logging.getLogger(__name__)


class AbstractModel(
    Generic[
        BatchInput,
        BaseOutput,
        ForwardOutput,
        EncodingInput,
        OptimizerInput,
        SchedulerInput,
        LearningRateInput,
    ],
    ABC,
    nn.Module,
):
    """Abstract base class for all models."""

    def __init__(self):
        """Initializes the model with device and fabric setup."""
        super(AbstractModel, self).__init__()
        nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self.fabric = Fabric(accelerator="cpu", precision="bf16-mixed")
        self.fabric.seed_everything(1234)

    @abstractmethod
    def get_optimizer(
        self,
        learning_rate: LearningRateInput,
        num_epochs: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> Tuple[OptimizerInput, SchedulerInput]:
        """
        Get the optimizer for the Modular-HIVAE.

        Args:
            learning_rate (LearningRateInput): Learning rate for the optimizer.
            num_epochs (int): Number of epochs for training.
            beta1 (float, optional): Beta1 hyperparameter for the Adam optimizer. Defaults to 0.9.
            beta2 (float, optional): Beta2 hyperparameter for the Adam optimizer. Defaults to 0.999.

        Returns:
            Tuple[OptimizerInput, SchedulerInput]: The optimizer and scheduler.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: BatchInput, mask: BatchInput) -> ForwardOutput:
        """
        Defines the computation performed at every call.

        Args:
            data (BatchInput): Input data for the forward pass.
            mask (BatchInput): Mask for the input data.

        Returns:
            ForwardOutput: Output of the forward pass.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, encoding: EncodingInput) -> BaseOutput:
        """
        Decodes the given encoding to the base output format.

        Args:
            encoding (EncodingInput): Encoding to be decoded.

        Returns:
            BaseOutput: Decoded output.
        """
        raise NotImplementedError

    @abstractmethod
    def _training_step(
        self, data: BatchInput, mask: BatchInput, optimizer: OptimizerInput
    ) -> float:
        """
        Perform a single training step.

        Args:
            data (BatchInput): Input data for the training step.
            mask (BatchInput): Mask for the input data.
            optimizer (OptimizerInput): Optimizer used for the training step.

        Returns:
            float: Loss value for the training step.
        """
        pass

    @staticmethod
    def _process_column(x_m_vt):
        """
        Processes a column of data.

        Args:
            x_m_vt: Tuple containing data, mask, and variable type.

        Returns:
            Processed column data.
        """
        x, m, vt = x_m_vt
        if vt.data_type == "cat":
            return Conversion._encode_categorical(x, m, vt.n_parameters)
        return x.view(-1, 1), m.view(-1, 1)

    @abstractmethod
    def _training_epoch(
        self, dataloader: DataLoader, optimizer: OptimizerInput
    ) -> Tuple[float, List[float]]:
        """
        Perform a single training epoch.

        Args:
            dataloader (DataLoader): DataLoader for training data.
            optimizer (OptimizerInput): Optimizer used for training.

        Returns:
            Tuple[float, List[float]]: Average loss and list of losses for each batch.
        """
        raise NotImplementedError

    @abstractmethod
    def _validation_step(self, data: BatchInput, mask: BatchInput) -> float:
        """
        Perform a single validation step.

        Args:
            data (BatchInput): Input data for the validation step.
            mask (BatchInput): Mask for the input data.

        Returns:
            float: Loss value for the validation step.
        """
        pass

    def _validation_epoch(self, dataloader: DataLoader) -> float:
        """
        Perform a validation epoch.

        Args:
            dataloader (DataLoader): DataLoader for validation data.

        Returns:
            float: Average validation loss.
        """
        self.eval()
        loss = []
        for data, missing in dataloader:
            loss.append(self._validation_step(data=data, mask=missing))

        # each loss is the average loss for a batch
        avg_loss = np.mean(loss)
        logger.info(f"Validation loss: {avg_loss}")
        return avg_loss

    @abstractmethod
    def fit(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        learning_rate: LearningRateInput,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, int]:
        """
        Fit the model to the training data.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train.
            learning_rate (LearningRateInput): Learning rate for the optimizer.
            val_dataloader (Optional[DataLoader], optional): DataLoader for validation data. Defaults to None.

        Returns:
            Tuple[float, int]: Best validation loss and number of epochs trained.
        """
        raise NotImplementedError

    @abstractmethod
    def _test_step(self, data: BatchInput, mask: BatchInput) -> float:
        """
        Perform a single test step.

        Args:
            data (BatchInput): Input data for the test step.
            mask (BatchInput): Mask for the input data.

        Returns:
            float: Loss value for the test step.
        """
        pass

    @abstractmethod
    def _predict_step(
        self, data: BatchInput, mask: BatchInput
    ) -> ForwardOutput:
        """
        Perform a single prediction step.

        Args:
            data (BatchInput): Input data for the prediction step.
            mask (BatchInput): Mask for the input data.

        Returns:
            ForwardOutput: Prediction output.
        """

        pass

    def predict(self, dataloader: DataLoader) -> ForwardOutput:
        """
        Perform prediction on a dataset.

        Args:
            dataloader (DataLoader): DataLoader for prediction data.

        Returns:
            ForwardOutput: Combined predictions for the entire dataset.

        Raises:
            Exception: If no data is provided to the model.
        """
        self.eval()
        outputs = None
        self = self.fabric.setup(self)
        dataloader = self.fabric.setup_dataloaders(dataloader)

        for data, mask in dataloader:
            tmp = self._predict_step(data=data, mask=mask)
            if outputs is None:
                outputs = tmp
            else:
                outputs += tmp

        if outputs is None:
            raise Exception("No data was provided to the model.")

        return outputs


class AbstractNormalModel(
    AbstractModel[
        NewBatchInput,
        NewBaseOutput,
        NewForwardOutput,
        EncodingInput,
        optim.Optimizer,
        optim.lr_scheduler._LRScheduler,
        float,
    ],
):
    def _training_epoch(
        self, dataloader: DataLoader, optimizer: optim.Optimizer
    ) -> Tuple[float, List[float]]:
        """Training epoch for HIVAE"""
        self.train()
        loss = []
        for data, missing in dataloader:
            ploss = self._training_step(
                data=data, mask=missing, optimizer=optimizer
            )
            loss.append(ploss)

        return np.mean(loss), loss

    def get_optimizer(
        self,
        learning_rate: float,
        num_epochs: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> Tuple[optim.Optimizer, optim.lr_scheduler.OneCycleLR]:
        """Get the optimizer for the Modular-HIVAE"""
        optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=0.01,
        )
        return optimizer, torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=learning_rate, total_steps=num_epochs
        )

    def fit(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        learning_rate: float,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Tuple[float, int]:
        """Fit the HIVAE model"""
        if num_epochs <= 0:
            raise Exception("Number of epochs must be at least 1")

        # determine number of trainable parameters and non-trainable parameters
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        non_trainable_params = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        logger.info(f"Trainable parameters: {trainable_params}")
        logger.info(f"Non-trainable parameters: {non_trainable_params}")

        optimizer, scheduler = self.get_optimizer(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
        )
        self, optimizer = self.fabric.setup(self, optimizer)
        self.fabric.clip_gradients(self, optimizer, max_norm=1.0)
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_dataloader is not None:
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        # use early stopping if val_dataloader is not None
        if val_dataloader is not None:
            best_loss = float("inf")
            patience = 0
        else:
            best_loss = None
            patience = None
        for current_epoch in tqdm(range(num_epochs), total=num_epochs):
            mlflow.log_metric("epoch", current_epoch, step=current_epoch)
            avg_loss, losses = self._training_epoch(
                dataloader=train_dataloader, optimizer=optimizer
            )
            mlflow.log_metric("train_loss", avg_loss, step=current_epoch)
            scheduler.step()

            if current_epoch % 10 == 0 or current_epoch == num_epochs - 1:
                logger.info(f"Loss at epoch {current_epoch}: {avg_loss}")
                print(f"Loss at epoch {current_epoch}: {avg_loss}")

            if val_dataloader is not None and current_epoch % 25 == 0:
                val_loss = self._validation_epoch(val_dataloader)
                mlflow.log_metric("val_loss", val_loss, step=current_epoch)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == 10:
                    logger.info("Early stopping")
                    break

        if val_dataloader is not None:
            val_loss = self._validation_epoch(val_dataloader)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0

        if best_loss is not None:
            best_epoch = (
                current_epoch - patience
                if patience is not None
                else current_epoch
            )
            if not isinstance(best_loss, float):
                best_loss = float(best_loss)
            assert isinstance(best_epoch, int)
        else:
            best_epoch = None

        return (
            best_loss,
            best_epoch,
        )


class AbstractModularModel(
    AbstractModel[
        NewBatchInput,
        NewBaseOutput,
        NewForwardOutput,
        EncodingInput,
        Tuple[optim.Optimizer, ...],
        Tuple[optim.lr_scheduler._LRScheduler, ...],
        Tuple[float, ...],
    ],
):
    """Abstract model class for normal models."""

    def _training_epoch(
        self,
        dataloader: DataLoader,
        optimizers: Tuple[optim.Optimizer, ...],
    ) -> Tuple[float, List[float]]:
        """
        Perform a single training epoch.

        Args:
            dataloader (DataLoader): DataLoader for training data.
            optimizer (optim.Optimizer): Optimizer used for training.

        Returns:
            Tuple[float, List[float]]: Average loss and list of losses for each batch.
        """
        self.train()
        losses = []
        for data, mask in dataloader:
            loss = self._training_step(
                data=data, mask=mask, optimizer=optimizers
            )
            losses.append(loss)
        return np.mean(losses), losses

    def get_optimizer(
        self,
        learning_rate: Tuple[float, ...],
        num_epochs: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> Tuple[Tuple[Optimizer, ...], Tuple[_LRScheduler, ...]]:
        """
        Get the optimizer for the model.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            num_epochs (int): Number of epochs for training.
            beta1 (float, optional): Beta1 hyperparameter for the Adam optimizer. Defaults to 0.9.
            beta2 (float, optional): Beta2 hyperparameter for the Adam optimizer. Defaults to 0.999.

        Returns:
            Tuple[optim.Optimizer, optim.lr_scheduler.OneCycleLR]: The optimizer and scheduler.
        """

        def _module_optimizer(module, learning_rate):
            opt = optim.Adam(
                module.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=0.01,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt, max_lr=learning_rate, total_steps=num_epochs
            )
            return opt, scheduler

        learning_rate, shared_learning_rate = (
            learning_rate[:-1],
            learning_rate[-1],
        )
        assert len(learning_rate) == len(self.module_models)
        out = tuple(
            _module_optimizer(module, lr)
            for module, lr in zip(self.module_models.values(), learning_rate)
        )
        optimizers = tuple(x[0] for x in out)
        schedulers = tuple(x[1] for x in out)

        # get the optimizer for the shared element
        shared_optimizer = (
            optim.Adam(
                self.shared_element.parameters(),
                lr=shared_learning_rate,
                betas=(beta1, beta2),
                weight_decay=0.01,
            )
            if self.shared_element.has_params
            else None
        )
        shared_scheduler = (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer=shared_optimizer,
                total_steps=num_epochs,
                max_lr=shared_learning_rate,
            )
            if shared_optimizer is not None
            else None
        )

        optimizers = (*optimizers, shared_optimizer)
        schedulers = (*schedulers, shared_scheduler)

        return optimizers, schedulers

    def fit(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        learning_rate: Tuple[float, ...],
        val_dataloader: DataLoader | None = None,
    ) -> Tuple[float, int]:
        """
        Fit the model to the training data.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
            val_dataloader (Optional[DataLoader], optional): DataLoader for validation data. Defaults to None.

        Returns:
            Tuple[float, int]: Best validation loss and number of epochs trained.

        Raises:
            Exception: If number of epochs is less than 1.
        """
        if num_epochs <= 0:
            raise Exception("Number of epochs must be at least 1")

        optimizers, schedulers = self.get_optimizer(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
        )

        if val_dataloader is not None:
            best_loss = float("inf")
            patience = 0
        else:
            best_loss = None
            patience = None

        for current_epoch in tqdm(range(num_epochs), total=num_epochs):
            mlflow.log_metric("epoch", current_epoch, step=current_epoch)
            avg_loss, losses = self._training_epoch(
                dataloader=train_dataloader, optimizers=optimizers
            )
            for scheduler, loss in zip(schedulers, losses):
                if scheduler is None:
                    continue
                scheduler.step()

            if val_dataloader is not None and (
                current_epoch % 10 == 0 or current_epoch == num_epochs - 1
            ):
                val_loss = self._validation_epoch(val_dataloader)
                mlflow.log_metric("val_loss", val_loss, step=current_epoch)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == 10:
                    logger.info("Early stopping")
                    break
        if val_dataloader is not None:
            val_loss = self._validation_epoch(val_dataloader)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0

        if best_loss is not None:
            best_epoch = (
                current_epoch - patience
                if patience is not None
                else current_epoch
            )
            if not isinstance(best_loss, float):
                best_loss = float(best_loss)
            assert isinstance(best_epoch, int)
        else:
            best_epoch = None

        return (best_loss, best_epoch)


class AbstractGanModel(
    AbstractModel[
        NewBatchInput,
        NewBaseOutput,
        NewForwardOutput,
        EncodingInput,
        Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer],
        Tuple[
            optim.lr_scheduler._LRScheduler,
            optim.lr_scheduler._LRScheduler,
            optim.lr_scheduler._LRScheduler,
        ],
        float,
    ]
):
    """Abstract model class for GAN models."""

    def _calc_gradient_penalty(
        self, real_data: Tensor, fake_data: Tensor
    ) -> Tensor:
        """
        Calculate the gradient penalty loss for WGAN-GP.

        Args:
            real_data (Tensor): Real data.
            fake_data (Tensor): Fake data.

        Returns:
            Tensor: Gradient penalty loss.
        """
        alpha = torch.rand(real_data.shape[0], 1, device=real_data.device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(
                disc_interpolates.size(), device=real_data.device
            ),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    @abstractmethod
    def _train_gan_discriminator_step(
        self,
        data: NewBatchInput,
        mask: NewBatchInput,
        optimizer: optim.Optimizer,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform a single GAN discriminator training step.

        Args:
            data (NewBatchInput): Input data for the discriminator.
            mask (NewBatchInput): Mask for the input data.
            optimizer (optim.Optimizer): Optimizer used for training.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Discriminator loss, real loss, and fake loss.
        """
        raise NotImplementedError

    @abstractmethod
    def _train_model_step(
        self,
        data: NewBatchInput,
        mask: NewBatchInput,
        optimizer: optim.Optimizer,
    ) -> NewForwardOutput:
        """
        Perform a single model training step.

        Args:
            data (NewBatchInput): Input data for the model.
            mask (NewBatchInput): Mask for the input data.
            optimizer (optim.Optimizer): Optimizer used for training.

        Returns:
            NewForwardOutput: Model output.
        """
        raise NotImplementedError

    @abstractmethod
    def _train_gan_generator_step(
        self,
        data: NewBatchInput,
        mask: NewBatchInput,
        optimizer: optim.Optimizer,
    ) -> Tensor:
        """
        Perform a single GAN generator training step.

        Args:
            data (NewBatchInput): Input data for the generator.
            mask (NewBatchInput): Mask for the input data.
            optimizer (optim.Optimizer): Optimizer used for training.

        Returns:
            Tensor: Generator loss.
        """
        raise NotImplementedError

    @abstractmethod
    def _train_model_from_discriminator_step(
        self,
        data: NewBatchInput,
        mask: NewBatchInput,
        optimizer: optim.Optimizer,
    ) -> Tensor:
        """
        Perform a single model training step from the discriminator.

        Args:
            data (NewBatchInput): Input data for the model.
            mask (NewBatchInput): Mask for the input data.
            optimizer (optim.Optimizer): Optimizer used for training.

        Returns:
            Tensor: Model loss from the discriminator.
        """
        raise NotImplementedError

    @staticmethod
    def concat_and_aggregate(
        metric_list: Tuple[Tensor, ...] | List[Tensor], n: int
    ) -> Tensor:
        """
        Concatenate and aggregate metric tensors.

        Args:
            metric_list (Tuple[Tensor, ...] | List[Tensor]): List of metric tensors.
            n (int): Number of items.

        Returns:
            Tensor: Aggregated metric tensor.
        """
        if metric_list[0].ndim == 0:
            metric_tensor = torch.stack(metric_list)
        else:
            metric_tensor = torch.cat(metric_list)
        return torch.sum(metric_tensor) / n

    @abstractmethod
    def _get_loss_from_output(self, output: NewForwardOutput) -> Tensor:
        """
        Get the loss from the model output.

        Args:
            output (NewForwardOutput): Model output.

        Returns:
            Tensor: Loss tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_number_of_items(self, mask: NewBatchInput) -> int:
        """
        Get the number of items from the mask.

        Args:
            mask (NewBatchInput): Mask for the input data.

        Returns:
            int: Number of items.
        """
        raise NotImplementedError

    def _training_epoch(
        self,
        dataloader: DataLoader,
        optimizers: Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer],
    ) -> Tuple[float, List[float]]:
        """
        Perform a single training epoch.

        Args:
            dataloader (DataLoader): DataLoader for training data.
            optimizers (Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer]): Optimizers used for training.

        Returns:
            Tuple[float, List[float]]: Average model loss and list of batch losses.
        """
        self.train()

        (
            model_optimizer,
            gan_discriminator_optimizer,
            gan_generator_optimizer,
        ) = optimizers

        # train the model
        loss = []
        items = []
        for data, missing in dataloader:
            ploss = self._get_loss_from_output(
                self._train_model_step(data, missing, model_optimizer)
            )
            loss.append(float(ploss.detach().cpu()))
            items.append(self._get_number_of_items(missing))

        items = sum(items)
        model_outputs = sum(loss) / items

        # train the GAN
        errD_loss = []
        errD_real_loss = []
        errD_fake_loss = []
        errG_loss = []
        errD_model_loss = []

        for data, missing in dataloader:
            errD, errD_real, errD_fake = self._train_gan_discriminator_step(
                data, missing, gan_discriminator_optimizer
            )
            errG = self._train_gan_generator_step(
                data, missing, gan_generator_optimizer
            )
            errD_ = self._train_model_from_discriminator_step(
                data, missing, model_optimizer
            )
            errD_loss.append(errD.detach().cpu())
            errD_real_loss.append(errD_real.detach().cpu())
            errD_fake_loss.append(errD_fake.detach().cpu())
            errG_loss.append(errG.detach().cpu())
            errD_model_loss.append(errD_.detach().cpu())

        # log the metrics
        mlflow.log_metric("train_model_loss", model_outputs)
        mlflow.log_metric(
            "train_errD_loss", self.concat_and_aggregate(errD_loss, items)
        )
        mlflow.log_metric(
            "train_errD_real_loss",
            self.concat_and_aggregate(errD_real_loss, items),
        )
        mlflow.log_metric(
            "train_errD_fake_loss",
            self.concat_and_aggregate(errD_fake_loss, items),
        )
        mlflow.log_metric(
            "train_errG_loss", self.concat_and_aggregate(errG_loss, items)
        )
        mlflow.log_metric(
            "train_errD_model_loss",
            self.concat_and_aggregate(errD_model_loss, items),
        )
        return model_outputs, loss

    def _get_optimizer(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer]:
        """
        Get the optimizers for the model, GAN discriminator, and GAN generator.

        Args:
            learning_rate (float): Learning rate for the optimizers.
            beta1 (float, optional): Beta1 hyperparameter for the Adam optimizer. Defaults to 0.9.
            beta2 (float, optional): Beta2 hyperparameter for the Adam optimizer. Defaults to 0.999.

        Returns:
            Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer]: Optimizers for the model, GAN discriminator, and GAN generator.
        """
        model_optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=0.01,
        )
        gan_discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=0.01,
        )
        gan_generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=0.01,
        )
        return (
            model_optimizer,
            gan_discriminator_optimizer,
            gan_generator_optimizer,
        )

    def get_optimizer(
        self,
        learning_rate: float,
        num_epochs: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> Tuple[
        Tuple[Optimizer, Optimizer, Optimizer],
        Tuple[_LRScheduler, _LRScheduler, _LRScheduler],
    ]:
        """
        Get the optimizers and schedulers for the model.

        Args:
            learning_rate (float): Learning rate for the optimizers.
            num_epochs (int): Number of epochs for training.
            beta1 (float, optional): Beta1 hyperparameter for the Adam optimizer. Defaults to 0.9.
            beta2 (float, optional): Beta2 hyperparameter for the Adam optimizer. Defaults to 0.999.

        Returns:
            Tuple[
                Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer],
                Tuple[optim.lr_scheduler._LRScheduler, optim.lr_scheduler._LRScheduler, optim.lr_scheduler._LRScheduler]
            ]: Optimizers and schedulers for the model, GAN discriminator, and GAN generator.
        """
        optimizers = self._get_optimizer(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
        )
        schedulers = tuple(
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=learning_rate,
                total_steps=num_epochs,
            )
            for optimizer in optimizers
        )
        return optimizers, schedulers

    def fit(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        learning_rate: float,
        val_dataloader: DataLoader | None = None,
    ) -> Tuple[float, int]:
        """
        Fit the model to the training data.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
            val_dataloader (Optional[DataLoader], optional): DataLoader for validation data. Defaults to None.

        Returns:
            Tuple[float, int]: Best validation loss and number of epochs trained.

        Raises:
            Exception: If number of epochs is less than 1.
        """
        if num_epochs <= 0:
            raise Exception("Number of epochs must be at least 1")

        # determine number of trainable parameters and non-trainable parameters
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        non_trainable_params = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        logger.info(f"Trainable parameters: {trainable_params}")
        logger.info(f"Non-trainable parameters: {non_trainable_params}")

        # get the optimizers and schedulers
        optimizers, schedulers = self.get_optimizer(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
        )
        out = self.fabric.setup(self, *optimizers)
        self = out[0]
        optimizers = out[1:]
        for optimizer in optimizers:
            self.fabric.clip_gradients(self, optimizer, max_norm=1.0)
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_dataloader is not None:
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

            # use early stopping if val_dataloader is not None
        if val_dataloader is not None:
            best_loss = float("inf")
            patience = 0
        else:
            best_loss = None
            patience = None

        for current_epoch in tqdm(range(num_epochs), total=num_epochs):
            mlflow.log_metric("epoch", current_epoch, step=current_epoch)
            avg_loss, losses = self._training_epoch(
                dataloader=train_dataloader, optimizers=optimizers
            )
            mlflow.log_metric("train_loss", avg_loss, step=current_epoch)
            for scheduler in schedulers:
                scheduler.step()

            if val_dataloader is not None and (
                current_epoch % 10 == 0 or current_epoch == num_epochs - 1
            ):
                val_loss = self._validation_epoch(val_dataloader)
                mlflow.log_metric("val_loss", val_loss, step=current_epoch)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == 10:
                    logger.info("Early stopping")
                    break
        if val_dataloader is not None:
            val_loss = self._validation_epoch(val_dataloader)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0

        if best_loss is not None:
            best_epoch = (
                current_epoch - patience
                if patience is not None
                else current_epoch
            )
            if not isinstance(best_loss, float):
                best_loss = float(best_loss)
            assert isinstance(best_epoch, int)
        else:
            best_epoch = None

        return (best_loss, best_epoch)


class AbstractGanModularModel(
    AbstractModel[
        NewBatchInput,
        NewBaseOutput,
        NewForwardOutput,
        EncodingInput,
        Tuple[Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer], ...],
        Tuple[
            Tuple[
                optim.lr_scheduler._LRScheduler,
                optim.lr_scheduler._LRScheduler,
                optim.lr_scheduler._LRScheduler,
            ],
            ...,
        ],
        Tuple[float, ...],
    ]
):
    """Abstract model class for GAN modular models."""

    def _calc_gradient_penalty(
        self, real_data: Tensor, fake_data: Tensor, discriminator: nn.Module
    ) -> Tensor:
        """
        Calculate the gradient penalty loss for WGAN-GP.

        Args:
            real_data (Tensor): Real data.
            fake_data (Tensor): Fake data.
            discriminator (nn.Module): Discriminator.

        Returns:
            Tensor: Gradient penalty loss.
        """
        alpha = torch.rand(real_data.shape[0], 1, device=real_data.device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(
                disc_interpolates.size(), device=real_data.device
            ),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    @abstractmethod
    def _train_model_step(
        self,
        data: NewBatchInput,
        mask: NewBatchInput,
        optimizer: Tuple[optim.Optimizer, ...],
    ) -> NewForwardOutput:
        """
        Perform a single model training step.

        Args:
            data (NewBatchInput): Input data for the model.
            mask (NewBatchInput): Mask for the input data.
            optimizer (Tuple[optim.Optimizer, ...]): Optimizers used for training.

        Returns:
            NewForwardOutput: Model output.
        """
        raise NotImplementedError

    @abstractmethod
    def _train_gan_discriminator_step(
        self,
        data: NewBatchInput,
        mask: NewBatchInput,
        optimizer: Tuple[optim.Optimizer, ...],
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], ...]:
        """
        Perform a single GAN discriminator training step.

        Args:
            data (NewBatchInput): Input data for the discriminator.
            mask (NewBatchInput): Mask for the input data.
            optimizer (Tuple[optim.Optimizer, ...]): Optimizers used for training.

        Returns:
            Tuple[Tuple[Tensor, Tensor, Tensor], ...]: Discriminator loss, real loss, and fake loss for each module.
        """
        raise NotImplementedError

    @abstractmethod
    def _train_gan_generator_step(
        self,
        data: NewBatchInput,
        mask: NewBatchInput,
        optimizer: Tuple[optim.Optimizer, ...],
    ) -> Tuple[Tensor, ...]:
        """
        Perform a single GAN generator training step.

        Args:
            data (NewBatchInput): Input data for the generator.
            mask (NewBatchInput): Mask for the input data.
            optimizer (Tuple[optim.Optimizer, ...]): Optimizers used for training.

        Returns:
            Tuple[Tensor, ...]: Generator loss for each module.
        """
        raise NotImplementedError

    @abstractmethod
    def _train_model_from_discriminator_step(
        self,
        data: NewBatchInput,
        mask: NewBatchInput,
        optimizer: Tuple[optim.Optimizer, ...],
    ) -> Tuple[Tensor, ...]:
        """
        Perform a single model training step from the discriminator.

        Args:
            data (NewBatchInput): Input data for the model.
            mask (NewBatchInput): Mask for the input data.
            optimizer (Tuple[optim.Optimizer, ...]): Optimizers used for training.

        Returns:
            Tuple[Tensor, ...]: Model loss from the discriminator for each module.
        """
        raise NotImplementedError

    @staticmethod
    def concat_and_aggregate(
        metric_list: Tuple[Tensor, ...] | List[Tensor], n: int
    ) -> Tensor:
        """
        Concatenate and aggregate metric tensors.

        Args:
            metric_list (Tuple[Tensor, ...] | List[Tensor]): List of metric tensors.
            n (int): Number of items.

        Returns:
            Tensor: Aggregated metric tensor.
        """
        return AbstractGanModel.concat_and_aggregate(metric_list, n)

    @abstractmethod
    def _get_loss_from_output(self, output: NewForwardOutput) -> float:
        """
        Get the loss from the model output.

        Args:
            output (NewForwardOutput): Model output.

        Returns:
            Tuple[Tensor, ...]: Loss tensor for each module.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_number_of_items(self, mask: NewBatchInput) -> int:
        """
        Get the number of items from the mask.

        Args:
            mask (NewBatchInput): Mask for the input data.

        Returns:
            Tuple[int, ...]: Number of items for each module.
        """
        raise NotImplementedError

    def _training_epoch(
        self,
        dataloader: DataLoader,
        optimizers: Tuple[
            Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer], ...
        ],
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform a single training epoch.

        Args:
            dataloader (DataLoader): DataLoader for training data.
            optimizers (Tuple[Tuple[optim.Optimizer, optim.Optimizer, optim.Optimizer], ...]): Optimizers used for training.

        Returns:
            Tuple[Tensor, Tensor]: Average model loss and list of batch losses.
        """
        self.train()
        losses = []
        items = []
        model_optimizers = tuple([opt[0] for opt in optimizers])
        gan_generator_optimizers = tuple([opt[1] for opt in optimizers])
        gan_discriminator_optimizers = tuple([opt[2] for opt in optimizers])
        # start = time.time()
        for data, mask in dataloader:
            ploss = self._get_loss_from_output(
                self._train_model_step(data, mask, model_optimizers)
            )
            losses.append(ploss)
            items.append(self._get_number_of_items(mask))
        losses = torch.tensor(losses)
        model_outputs = losses.sum() / torch.stack(items).sum()
        # print(f"Model training time: {time.time() - start}")

        # train the GAN
        errD_loss = []
        errD_real_loss = []
        errD_fake_loss = []
        errG_loss = []
        errD_model_loss = []

        # start = time.time()
        for data, mask in dataloader:
            errD, errD_real, errD_fake = self._train_gan_discriminator_step(
                data, mask, gan_discriminator_optimizers
            )
            errG = self._train_gan_generator_step(
                data, mask, gan_generator_optimizers
            )
            errD_ = self._train_model_from_discriminator_step(
                data, mask, model_optimizers
            )
            errD_loss.append(errD)
            errD_real_loss.append(errD_real)
            errD_fake_loss.append(errD_fake)
            errG_loss.append(errG)
            errD_model_loss.append(errD_)

        # print(f"GAN training time: {time.time() - start}")

        # stack the losses
        errD_per_module = torch.tensor(errD_loss).sum(dim=0)
        errD_real_per_module = torch.tensor(errD_real_loss).sum(dim=0)
        errD_fake_per_module = torch.tensor(errD_fake_loss).sum(dim=0)
        errG_per_module = torch.tensor(errG_loss).sum(dim=0)
        errD_model_per_module = torch.tensor(errD_model_loss).sum(dim=0)

        # log the summed metrics
        mlflow.log_metric("train_model_loss", model_outputs)
        mlflow.log_metric("train_errD_loss", errD_per_module.sum())
        mlflow.log_metric("train_errD_real_loss", errD_real_per_module.sum())
        mlflow.log_metric("train_errD_fake_loss", errD_fake_per_module.sum())
        mlflow.log_metric("train_errG_loss", errG_per_module.sum())
        mlflow.log_metric("train_errD_model_loss", errD_model_per_module.sum())

        return model_outputs, losses

    def get_optimizer(
        self,
        learning_rate: Tuple[float, ...],
        num_epochs: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> Tuple[
        Tuple[Tuple[Optimizer, Optional[Optimizer], Optional[Optimizer]], ...],
        Tuple[_LRScheduler, ...],
    ]:
        """
        Get the optimizers and schedulers for the model.

        Args:
            learning_rate (Tuple[float, ...]): Learning rates for the optimizers.
            num_epochs (int): Number of epochs for training.
            beta1 (float, optional): Beta1 hyperparameter for the Adam optimizer. Defaults to 0.9.
            beta2 (float, optional): Beta2 hyperparameter for the Adam optimizer. Defaults to 0.999.

        Returns:
            Tuple[
                Tuple[Tuple[optim.Optimizer, Optional[optim.Optimizer], Optional[optim.Optimizer]], ...],
                Tuple[optim.lr_scheduler._LRScheduler, ...]
            ]: Optimizers and schedulers for the model, GAN discriminator, and GAN generator.
        """

        def _module_optimizer(module, learning_rate):
            opt = optim.Adam(
                module.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=0.01,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt, max_lr=learning_rate, total_steps=num_epochs
            )
            return opt, scheduler

        learning_rate, shared_learning_rate = (
            learning_rate[:-1],
            learning_rate[-1],
        )
        assert len(learning_rate) == len(self.model.module_models)
        out = tuple(
            _module_optimizer(module, lr)
            for module, lr in zip(
                self.model.module_models.values(), learning_rate
            )
        )
        model_optimizers = tuple(x[0] for x in out)
        model_schedulers = tuple(x[1] for x in out)

        # get the optimizer for the shared element
        shared_optimizer = (
            optim.Adam(
                self.model.shared_element.parameters(),
                lr=shared_learning_rate,
                betas=(beta1, beta2),
                weight_decay=0.01,
            )
            if self.model.shared_element.has_params
            else None
        )
        shared_scheduler = (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer=shared_optimizer,
                total_steps=num_epochs,
                max_lr=shared_learning_rate,
            )
            if shared_optimizer is not None
            else None
        )

        model_optimizers = (*model_optimizers, shared_optimizer)
        schedulers = (*model_schedulers, shared_scheduler)

        assert isinstance(learning_rate, Tuple), "Learning rate must be a tuple"
        # Get the GAN optimizers
        gan_discriminator_optimizers = tuple(
            optim.Adam(
                module.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=0.01,
            )
            for lr, module in zip(learning_rate, self.discriminators)
        )

        gan_generator_optimizers = tuple(
            optim.Adam(
                module.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=0.01,
            )
            for lr, module in zip(learning_rate, self.generators)
        )

        optimizers = [
            (mod, gen, disc)
            for mod, gen, disc in zip(
                model_optimizers,
                gan_generator_optimizers,
                gan_discriminator_optimizers,
            )
        ]
        optimizers.append((shared_optimizer, None, None))

        return optimizers, schedulers  # type: ignore

    def fit(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        learning_rate: Tuple[float],
        val_dataloader: DataLoader | None = None,
    ) -> Tuple[float, int]:
        """
        Fit the model to the training data.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train.
            learning_rate (Tuple[float]): Learning rates for the optimizers.
            val_dataloader (Optional[DataLoader], optional): DataLoader for validation data. Defaults to None.

        Returns:
            Tuple[float, int]: Best validation loss and number of epochs trained.

        Raises:
            Exception: If number of epochs is less than 1.
        """

        if num_epochs <= 0:
            raise Exception("Number of epochs must be at least 1")

        # determine number of trainable parameters and non-trainable parameters
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        non_trainable_params = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        logger.info(f"Trainable parameters: {trainable_params}")
        logger.info(f"Non-trainable parameters: {non_trainable_params}")

        # get the optimizers and schedulers
        optimizers, schedulers = self.get_optimizer(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
        )
        flattened_optimizers = [
            opt
            for optimizer in optimizers
            for opt in optimizer
            if opt is not None
        ]
        out = self.fabric.setup(self, *flattened_optimizers)
        self = out[0]
        flattened_optimizers = list(out[1:])
        # append two None values to the end of the list to match the length of the optimizers
        for optimizer in flattened_optimizers:
            self.fabric.clip_gradients(self, optimizer, max_norm=1.0)
        flattened_optimizers.extend(
            [None, None] if self.model.shared_element.has_params else [None] * 3
        )
        optimizers = [
            tuple(flattened_optimizers[i : i + 3])
            for i in range(0, len(flattened_optimizers), 3)
        ]

        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_dataloader is not None:
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        # use early stopping if val_dataloader is not None
        if val_dataloader is not None:
            best_loss = float("inf")
            patience = 0
        else:
            best_loss = None
            patience = None

        for current_epoch in tqdm(range(num_epochs), total=num_epochs):
            mlflow.log_metric("epoch", current_epoch, step=current_epoch)
            avg_loss, losses = self._training_epoch(
                dataloader=train_dataloader, optimizers=optimizers
            )
            mlflow.log_metric("train_loss", avg_loss, step=current_epoch)
            for scheduler in schedulers:
                if scheduler is None:
                    continue
                scheduler.step()

            if val_dataloader is not None and (
                current_epoch % 10 == 0 or current_epoch == num_epochs - 1
            ):
                val_loss = self._validation_epoch(val_dataloader)
                mlflow.log_metric("val_loss", val_loss, step=current_epoch)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == 10:
                    logger.info("Early stopping")
                    break
        if val_dataloader is not None:
            val_loss = self._validation_epoch(val_dataloader)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0

        if best_loss is not None:
            best_epoch = (
                current_epoch - patience
                if patience is not None
                else current_epoch
            )
            if not isinstance(best_loss, float):
                best_loss = float(best_loss)
            assert isinstance(best_epoch, int)
        else:
            best_epoch = None

        return (best_loss, best_epoch)
