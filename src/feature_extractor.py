import logging
from torch import nn
import torchaudio
import torch
import numpy as np

#region Logging


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

class SpectrogramExtractor(nn.Module):
    """
    This class is a feature extractor that uses the MelSpectrogram transformation from torchaudio.

    The output of the feature extractor is a tensor with the shape [batch_size, time, mel_bands]. 
    """

    def __init__(self, sample_rate, feature_extractor_output_vectors_dimension) -> None:
        super().__init__()

        self.init_feature_extractor(sample_rate, feature_extractor_output_vectors_dimension)

    def init_feature_extractor(self, sample_rate, feature_extractor_output_vectors_dimension):
        # Implement your feature extractor initialization code here

        # TODO: Ask Javier about these magic things.
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            n_fft = 2048,
            win_length = int(sample_rate * 0.025),
            hop_length = int(sample_rate * 0.01),
            n_mels = feature_extractor_output_vectors_dimension,
            mel_scale = "slaney",
            window_fn = torch.hamming_window,
            f_max = sample_rate // 2,
            center = False,
            normalized = False,
            norm = "slaney",
        )
    
    def extract_features(self, audio_signal):
        features = self.feature_extractor(audio_signal)

        # HACK it seems that the feature extractor output spectrogram has mel bands as rows and time as columns
        features = features.transpose(1, 2)

        return features
    
    def __call__(self, waveforms):
        logger.debug(f"waveforms shape: {waveforms.shape}")
        logger.debug(f"wavefor.size(): {waveforms.size()}")
        features = self.extract_features(waveforms)
        logger.debug(f"features.size(): {features.size()}")
        return features