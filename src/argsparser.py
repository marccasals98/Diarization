import argparse

class ArgsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Speaker Diarization test.')
    
    def parse_args(self):
        self.parser.add_argument(
            '--audio_path',
            type=str,
            default='/gpfs/projects/bsc88/speech/data/raw_data/diarization/voxconverse/audio/dev/audio',
        )
        self.parser.add_argument(
            '--rttm_path',
            type=str,
            default="/gpfs/projects/bsc88/speech/data/raw_data/diarization/voxconverse/dev",
        )
        self.parser.add_argument(
            '--segment_length',
            type=int,
            default=5,
        )
        self.parser.add_argument(
            '--feature_stride',
            type=float,
            default=0.01,
        )
        self.parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
        )
        self.parser.add_argument(
            '--allow_overlap',
            type=bool,
            default=True,
        )
        self.parser.add_argument(
            '--max_num_speakers',
            type=int,
            default=20,
        )
        self.parser.add_argument(
            '--frame_length',
            type=float,
            default=0.025,
        )
        self.parser.add_argument(
            '--num_workers',
            type=int,
            default=4,
        )
        self.parser.add_argument(
            '--optimizer',
            type=str,
            default='adam',
        )
        self.parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.001,
        )
        self.parser.add_argument(
            '--feature_extractor',
            type=str,
            default="SpectrogramExtractor",
        )
        self.parser.add_argument(
            '--weight_decay',
            type=float,
            default=0.01,
        )
        self.parser.add_argument(
            '--load_checkpoint',
            type=bool,
            default=False,
        )
        self.parser.add_argument(
            '--max_epochs',
            type=int,
            default=5,
        )
        self.parser.add_argument(
            '--feature_extractor_output_vectors_dimension',
            type=int,
            default=150,
        )
        self.parser.add_argument(
            '--n_fft',
            type=int,
            default=2048,
        )
        self.parser.add_argument(
            '--sample_rate',
            type=int,
            default=16000,
        )
        self.parser.add_argument(
            '--detach_attractor_loss',
            type=bool,
            default=False,
        )
        return self.parser.parse_args()
    
