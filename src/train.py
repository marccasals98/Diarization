from argsparser import ArgsParser
import logging
from data import TrainDataset
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from eend import BLSTM_EEND
from torch import nn
from losses import PITLoss
import datetime
import os
from tools import get_memory_info


# region logging
# Logging
# -------
# Set logging config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
)

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
# endregion


class Trainer:
    def __init__(self, params) -> None:
        self.start_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        self.params = params
        logger.info(self.params)
        self.init_training_variables()
        self.set_device()
        self.set_random_seed()
        self.load_training_data()
        self.load_validation_data()
        self.load_network()
        self.load_loss_function()
        self.load_optimizer()
        self.main()

    # region initialization
    def init_training_variables(self):
        self.step = 0
        self.epoch = 0
        self.train_loss = 0
        self.best_train_loss = np.inf
        self.best_validation_loss = np.inf
        self.validation_eval_metric = 0.0
        self.training_eval_metric = 0.0
        self.best_model_validation_eval_metric = 0.0
        self.best_model_training_eval_metric = 0.0
        self.validations_without_improvement = 0.0
        self.validations_without_improvement_or_opt_update = 0.0
        self.early_stopping_flag = False

    def set_device(self):
        """Set torch device."""
        logger.info("Setting device...")
        # Set device to GPU or CPU depending on what is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running on {self.device} device.")
        if self.device == "cuda":
            self.gpus_count = torch.cuda.device_count()
            logger.info(f"{self.gpus_count} GPUs available.")
            # Batch size should be divisible by number of GPUs
        else:
            self.gpus_count = 0

        logger.info("Device setted.")

    def set_random_seed(self):

        logger.info("Setting random seed...")

        random.seed(1234)
        np.random.seed(1234)

        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        torch.backends.cudnn.deterministic = True

        logger.info("Random seed setted.")

    # endregion

    # region Loaders
    def load_network(self):
        logger.info("Loading the network...")

        self.net = BLSTM_EEND(
            self.params.segment_length,
            self.params.frame_length,
            self.params.feature_extractor,
            self.params.sample_rate,
            self.params.feature_extractor_output_vectors_dimension,
            self.params.max_num_speakers,
            self.params.dropout,
            self.params.hidden_size,
            self.params.n_layers,
            self.params.embedding_layers,
            self.params.embedding_size,
            self.params.dc_loss_ratio,
            self.params.n_fft,
        )

        # Data Parallelism
        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
            logger.info("Data Parallelism enabled.")
        self.net.to(self.device)

        # Print the number of trainable parameters
        self.total_trainable_params = 0
        parms_dict = {}

        logger.info(f"Detail of every trainable layer:")
        for name, parameter in self.net.named_parameters():

            layer_name = name.split(".")[1]
            if layer_name not in parms_dict.keys():
                parms_dict[layer_name] = 0

            logger.debug(f"name: {name}, layer_name: {layer_name}")

            if not parameter.requires_grad:
                continue
            trainable_params = parameter.numel()

            logger.info(f"{name} is trainable with {parameter.numel()} parameters")

            parms_dict[layer_name] = parms_dict[layer_name] + trainable_params

            self.total_trainable_params += trainable_params

        # Check if this is correct
        logger.info(
            f"Total trainable parameters per layer:{self.total_trainable_params}"
        )
        for layer_name in parms_dict.keys():
            logger.info(f"{layer_name}: {parms_dict[layer_name]}")

        # summary(self.net, (150, self.params.feature_extractor_output_vectors_dimension))

        logger.info(
            f"Network loaded, total_trainable_params: {self.total_trainable_params}"
        )

        logger.info("Network loaded.")

    def info_mem(self, step = None, logger_level = "INFO"):

        '''Logs CPU and GPU free memory.'''
        
        cpu_available_pctg, gpu_free = get_memory_info()
        if step is not None:
            message = f"Step {self.step}: CPU available {cpu_available_pctg:.2f}% - GPU free {gpu_free}"
        else:
            message = f"CPU available {cpu_available_pctg:.2f}% - GPU free {gpu_free}"
        
        if logger_level == "INFO":
            logger.info(message)
        elif logger_level == "DEBUG":
            logger.debug(message)

    def load_loss_function(self):
        self.loss_function = PITLoss(
            n_speakers=self.params.max_num_speakers,
            detach_attractor_loss=self.params.detach_attractor_loss,
        )
        logger.info("Loss function loaded.")

    def load_optimizer(self):
        logger.info("Loading the optimizer...")

        if self.params.optimizer == "adam":
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay,
            )
        if self.params.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(
                # self.net.parameters(),
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay,
            )
        if self.params.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                # self.net.parameters(),
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay,
            )

        if self.params.load_checkpoint == True:
            self.load_checkpoint_optimizer()
        logger.info(f"Optimizer {self.params.optimizer} loaded!")

    def compute_number_of_frames(self):
        segment_samples = int(self.params.segment_length * self.params.sample_rate)
        if self.params.feature_extractor == "SpectrogramExtractor":
            n_frames = (
                int(segment_samples - self.params.n_fft)
                // int(self.params.sample_rate * self.params.feature_stride)
                + 1
            )
        else:
            logger.info("Feature extractor not implemented.")
        return n_frames

    def load_training_data(self):
        """Loads the training data and generate the DataLoader."""
        n_frames = self.compute_number_of_frames()

        training_dataset = TrainDataset(
            audio_files=self.params.audio_path_train,
            rttm_paths=self.params.rttm_path_train,
            feature_stride=self.params.feature_stride,
            n_frames=n_frames,
            segment_length=self.params.segment_length,
            allow_overlap=self.params.allow_overlap,
            max_num_speakers=self.params.max_num_speakers,
            frame_length=self.params.frame_length,
        )
        data_loader_parameters = {
            "batch_size": self.params.batch_size,
            "shuffle": True,
            "num_workers": self.params.num_workers,
        }
        self.training_generator = DataLoader(training_dataset, **data_loader_parameters)
        self.total_batches = len(self.training_generator)
        logger.info("Training data loaded.")
        del training_dataset

    def load_validation_data(self):
        """Loads the validation data and generate the DataLoader."""
        n_frames = self.compute_number_of_frames()

        validation_dataset = TrainDataset(
            audio_files=self.params.audio_path_validation,
            rttm_paths=self.params.rttm_path_validation,
            feature_stride=self.params.feature_stride,
            n_frames=n_frames,
            segment_length=self.params.segment_length,
            allow_overlap=self.params.allow_overlap,
            max_num_speakers=self.params.max_num_speakers_validation,
            frame_length=self.params.frame_length,
        )
        data_loader_parameters = {
            "batch_size": self.params.batch_size,
            "shuffle": True,
            "num_workers": self.params.num_workers,
        }
        self.validation_generator = DataLoader(validation_dataset, **data_loader_parameters)
        self.total_batches = len(self.validation_generator)
        logger.info("Validation data loaded.")
        del validation_dataset

    # endregion

    #region Training Pipeline
    def check_print_training_info(self):
        if self.step > 0 and self.params.print_training_info_every > 0 \
            and self.step % self.params.print_training_info_every == 0:

            info_to_print = f"Epoch {self.epoch} of {self.params.max_epochs}, "
            info_to_print = info_to_print + f"batch {self.current_batch} of {self.total_batches}, "
            info_to_print = info_to_print + f"step {self.step}, "
            info_to_print = info_to_print + f"Loss {self.train_loss:.3f}, "
            info_to_print = info_to_print + f"Best training score:{self.best_model_training_eval_metric:.3f},"
            # info_to_print = info_to_print + f"Best validation score: {self.best_model_validation_eval_metric:.3f}"

            logger.info(info_to_print)

            # Uncomment for memory usage info 
            self.info_mem(self.step, logger_level = "DEBUG")
    def check_early_stopping(self):
        """
        Check if we have to do early stopping. 
        If the conditions are met, self.early_stopping_flag = True
        """

        if self.params.early_stopping > 0 \
            and self.validations_without_improvement >= self.params.early_stopping:

            self.early_stopping_flag = True
            logger.info(f"Doing early stopping after {self.validations_without_improvement} validations without improvement")
    def eval_and_save_best_model(self):

        if (
            self.step > 0
            and self.params.eval_and_save_best_model_every > 0
            and self.step % self.params.eval_and_save_best_model_every == 0
        ):

            logger.info("Evaluating and saving the new best model (if founded)...")

            # Calculate the evaluation metrics
            self.evaluate()

            # Have we found a better model? (Better in validation metric).
            if self.validation_eval_metric > self.best_model_validation_eval_metric:

                logger.info("We found a better model!")

                # Update best model evaluation metrics
                self.best_model_train_loss = self.train_loss
                self.best_model_training_eval_metric = self.training_eval_metric
                self.best_model_validation_eval_metric = self.validation_eval_metric

                logger.info(f"Best model train loss: {self.best_model_train_loss:.3f}")
                logger.info(
                    f"Best model train evaluation metric: {self.best_model_training_eval_metric:.3f}"
                )
                logger.info(
                    f"Best model validation evaluation metric: {self.best_model_validation_eval_metric:.3f}"
                )

                self.save_model()

                # Since we found and improvement, validations_without_improvement and validations_without_improvement_or_opt_update are reseted.
                self.validations_without_improvement = 0
                self.validations_without_improvement_or_opt_update = 0

            else:
                # In this case the search didn't improved the model
                # We are one validation closer to do early stopping
                self.validations_without_improvement = (
                    self.validations_without_improvement + 1
                )
                self.validations_without_improvement_or_opt_update = (
                    self.validations_without_improvement_or_opt_update + 1
                )

            logger.info(
                f"Consecutive validations without improvement: {self.validations_without_improvement}"
            )
            logger.info(
                f"Consecutive validations without improvement or optimizer update: {self.validations_without_improvement_or_opt_update}"
            )
            logger.info("Evaluating and saving done.")
            self.info_mem(self.step, logger_level="DEBUG")

    def train_single_epoch(self, epoch):
        logger.info(f"Training epoch {epoch+1} of {self.params.max_epochs}")

        self.net.train()
        for self.current_batch, batch_data in enumerate(self.training_generator):

            logger.info(f"step: {self.step}")
            input, label = batch_data

            # Assign data to device
            input, label = input.to(self.device), label.to(self.device)
            n_speakers = np.asarray(
                [
                    max(torch.where(t.sum(0).cpu() != 0)[0]) + 1 if t.sum() > 0 else 0
                    for t in label
                ]
            )
            if self.current_batch == 0:
                logger.info(f"input.shape: {input.shape}")
                logger.info(f"label.shape: {label.shape}")

            _, prediction, embeddings = self.net(input)

            self.loss = self.loss_function(prediction, label, n_speakers)
            self.train_loss = self.loss.item()

            # BACKPROPAGATION

            # Zero the gradients
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            self.loss.backward()

            # Update the weights
            self.optimizer.step()

            self.eval_and_save_best_model()

            self.check_print_training_info()

            # Update best loss
            if self.train_loss < self.best_train_loss:
                self.best_train_loss = self.train_loss

            self.step += 1

        logger.info(f"-"*50)
        logger.info(f"Epoch {epoch} finished with:")
        logger.info(f"Loss {self.train_loss:.3f}")
        logger.info(f"Best model training evaluation metric: {self.best_model_training_eval_metric:.3f}")
        logger.info(f"Best model validation evaluation metric: {self.best_model_validation_eval_metric:.3f}")
        logger.info(f"-"*50)



    def evaluate_training(self):
        logger.info("Evaluating training data...")
        self.net.eval()
        with torch.no_grad():
            for self.current_batch, batch_data in enumerate(self.training_generator):
                input, label = batch_data

                # Assign data to device
                input, label = input.to(self.device), label.to(self.device)
                n_speakers = np.asarray(
                    [
                        max(torch.where(t.sum(0).cpu() != 0)[0]) + 1 if t.sum() > 0 else 0
                        for t in label
                    ]
                )

                _, prediction, embeddings = self.net(input)

                self.loss = self.loss_function(prediction, label, n_speakers)
                self.train_loss = self.loss.item()

                # Update best loss
                if self.train_loss < self.best_train_loss:
                    self.best_train_loss = self.train_loss
    def evaluate_validation(self):
        logger.info("Evaliating validation data...")
        self.net.eval()
        with torch.no_grad():
            for self.current_batch, batch_data in enumerate(self.validation_generator):
                input, label = batch_data

                # Assign data to device
                input, label = input.to(self.device), label.to(self.device)
                n_speakers = np.asarray(
                    [
                        max(torch.where(t.sum(0).cpu() != 0)[0]) + 1 if t.sum() > 0 else 0
                        for t in label
                    ]
                )

                _, prediction, embeddings = self.net(input)

                self.loss = self.loss_function(prediction, label, n_speakers)
                self.validation_loss = self.loss.item()


                # Update best loss
                if self.validation_loss < self.best_validation_loss:
                    self.best_validation_loss = self.validation_loss
                    self.best_model_validation_eval_metric = self.validation_eval_metric
                    self.best_model_training_eval_metric = self.training_eval_metric
    def evaluate(self):
        self.evaluate_training()
        self.evaluate_validation()

    def save_model(self):

        '''Function to save the model info and optimizer parameters.'''

        # 1 - Add all the info that will be saved in checkpoint 
        
        model_results = {
            'best_model_train_loss' : self.best_model_train_loss,
            'best_model_training_eval_metric' : self.best_model_training_eval_metric,
            'best_model_validation_eval_metric' : self.best_model_validation_eval_metric,
        }

        training_variables = {
            'epoch': self.epoch,
            'current_batch' : self.current_batch,
            'step' : self.step,
            'validations_without_improvement' : self.validations_without_improvement,
            'validations_without_improvement_or_opt_update' : self.validations_without_improvement_or_opt_update,
            'train_loss' : self.train_loss,
            'best_train_loss' : self.best_train_loss,
            'best_model_train_loss' : self.best_model_train_loss,
            'best_model_training_eval_metric' : self.best_model_training_eval_metric,
            'best_model_validation_eval_metric' : self.best_model_validation_eval_metric,
            'total_trainable_params' : self.total_trainable_params,
        }
        
        if torch.cuda.device_count() > 1:
            checkpoint = {
                'model': self.net.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'settings': self.params,
                'model_results' : model_results,
                'training_variables' : training_variables,
                }
        else:
            checkpoint = {
                'model': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'settings': self.params,
                'model_results' : model_results,
                'training_variables' : training_variables,
                }


        end_datetime = datetime.datetime.strftime(datetime.datetime.now(), '%y-%m-%d %H:%M:%S')
        checkpoint['start_datetime'] = self.start_datetime
        checkpoint['end_datetime'] = end_datetime

        # 2 - Save the checkpoint locally

        checkpoint_folder = os.path.join(self.params.save_model_path, self.params.model_name)
        checkpoint_file_name = f"{self.params.model_name}.chkpt"
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file_name)

        # Create directory if doesn't exists
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        
        # Save the model parameters in a json file
        logger.info(f"Saving trainable parameters in a json file...")
        self.save_trainable_params_json(self.net, self.params.json_output_folder)
        logger.info(f"The json was saved in {self.params.json_output_folder}")
        
        
        logger.info(f"Saving training and model information in {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Done.")

        # Delete variables to free memory
        del model_results
        del training_variables
        del checkpoint

        logger.info(f"Training and model information saved.")

    def train(self):
        for self.epoch in range(self.params.max_epochs):
            self.train_single_epoch(self.epoch)

    def main(self):
        self.train()
def main():
    args = ArgsParser().parse_args()
    Trainer(args)


if __name__ == "__main__":
    main()
