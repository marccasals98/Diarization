from argsparser import ArgsParser
import logging
from data import TrainDataset

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
    def __init__(self, parameters)->None:
        self.parameters = parameters
        print(self.parameters)
        self.load_training_data()
        self.main()

    def load_training_data(self):
        training_dataset = TrainDataset(parameters=self.parameters)

    def train_single_epoch(self, epoch):
        logger.info(f"Training epoch {epoch+1} of {self.parameters.max_epochs}")

    def train(self):
        for self.epoch in range(self.parameters.max_epochs):
            self.train_single_epoch(self.epoch)

    def main(self):
        self.train()

def main():
    args = ArgsParser().parse_args()
    Trainer(args)

if __name__ == '__main__':
    main()