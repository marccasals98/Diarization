from argsparser import ArgsParser
import logging
from data import TrainDataset
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from eend import EEND_Model
from torch import nn
#region logging
# Logging
# -------
# Set logging config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_formatter = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%y-%m-%d %H:%M:%S',
    )

# Set a logging stream handler
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setLevel(logging.INFO)
logger_stream_handler.setFormatter(logger_formatter)

# Add handlers
logger.addHandler(logger_stream_handler)
#endregion

class Trainer:
    def __init__(self, params)->None:
        self.params = params
        print(self.params)
        self.set_device()
        self.set_random_seed()
        self.load_training_data()
        self.load_network()
        self.load_loss_function()
        self.load_optimizer()
        self.main()

    #region initialization
    def set_device(self):
        '''Set torch device.'''
        logger.info('Setting device...')
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
    #endregion
    
    #region Loaders
    def load_network(self):
        logger.info("Loading the network...")

        self.net = EEND_Model(self.params, self.device)

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
        logger.info(f"Total trainable parameters per layer:{self.total_trainable_params}")
        for layer_name in parms_dict.keys():
            logger.info(f"{layer_name}: {parms_dict[layer_name]}")

        #summary(self.net, (150, self.params.feature_extractor_output_vectors_dimension))

        logger.info(f"Network loaded, total_trainable_params: {self.total_trainable_params}")

        logger.info("Network loaded.")
    def load_loss_function(self):
        ...
    def load_optimizer(self):
        logger.info("Loading the optimizer...")

        if self.params.optimizer == 'adam':
            logger.info(f"self.net.parameters(): {self.net.parameters()}")
            logger.info(f"self.net.parameters(): {list(self.net.parameters())}")
            logger.info(f"self.net.parameters(): {list(filter(lambda p: p.requires_grad, self.net.parameters()))}")
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.net.parameters()),
                lr=self.params.learning_rate,
                weight_decay=self.params.weight_decay
            )
        if self.params.optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(
                #self.net.parameters(), 
                filter(lambda p: p.requires_grad, self.net.parameters()), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
                )
        if self.params.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                #self.net.parameters(), 
                filter(lambda p: p.requires_grad, self.net.parameters()), 
                lr=self.params.learning_rate, 
                weight_decay=self.params.weight_decay,
                )       
            
        if self.params.load_checkpoint == True:
            self.load_checkpoint_optimizer()
        logger.info(f"Optimizer {self.params.optimizer} loaded!")
        
    def load_training_data(self):
        """Loads the training data and generate the DataLoader.
        """
        training_dataset = TrainDataset(
            audio_files=self.params.audio_path,
            rttm_paths=self.params.rttm_path,
            segment_length=self.params.segment_length,
            allow_overlap=self.params.allow_overlap,
            max_num_speakers=self.params.max_num_speakers,
            frame_length=self.params.frame_length,
        )
        data_loader_parameters = {
            'batch_size': self.params.batch_size,
            'shuffle': True,
            'num_workers': self.params.num_workers,
        }
        self.training_generator = DataLoader(
            training_dataset,
            **data_loader_parameters
        )
        logger.info("Training data loaded.")
        del training_dataset
    
    #endregion
    def train_single_epoch(self, epoch):
        logger.info(f"Training epoch {epoch+1} of {self.params.max_epochs}")

    def train(self):
        for self.epoch in range(self.params.max_epochs):
            self.train_single_epoch(self.epoch)

    def main(self):
        self.train()

def main():
    args = ArgsParser().parse_args()
    Trainer(args)

if __name__ == '__main__':
    main()