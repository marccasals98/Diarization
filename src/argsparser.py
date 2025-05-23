import argparse

class ArgsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Speaker Diarization test.')
    
    def parse_args(self):
        self.parser.add_argument(
            '--audio_path_train',
            type=str,
            default='/gpfs/projects/bsc88/speech/data/raw_data/diarization/voxconverse/audio/dev/audio',
        )
        self.parser.add_argument(
            '--rttm_path_train',
            type=str,
            default="/gpfs/projects/bsc88/speech/data/raw_data/diarization/voxconverse/dev",
        )
        self.parser.add_argument(
            '--audio_path_validation',
            type=str,
            default='/gpfs/projects/bsc88/speech/data/raw_data/diarization/voxconverse/audio/voxconverse_test_wav',
        )
        self.parser.add_argument(
            '--rttm_path_validation',
            type=str,
            default="/gpfs/projects/bsc88/speech/data/raw_data/diarization/voxconverse/test",
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
            '--print_training_info_every',
            type=int,
            default=50,
        )
        self.parser.add_argument(
            '--eval_and_save_best_model_every',
            type=int,
            default=500,
        )
        self.parser.add_argument(
            '--save_model_path',
            type=str,
            default='/gpfs/projects/bsc88/speech/speaker_recognition/outputs/diarization/models',
        )
        self.parser.add_argument(
            '--model_name',
            type=str,
            default='debug_model',
        )
        self.parser.add_argument(
            '--allow_overlap',
            type=bool,
            default=True,
        )
        self.parser.add_argument(
            '--max_num_speakers',
            type=int,
            default=21,
        )
        self.parser.add_argument(
            '--max_num_speakers_validation',
            type=int,
            default=21,
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
            default=20,
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
        self.parser.add_argument(
            '--dropout',
            type=float,
            default=0.25,
        )
        self.parser.add_argument(
            '--hidden_size',
            type=int,
            default=256,
        )
        self.parser.add_argument(
            '--n_layers',
            type=int,
            default=20,
        )
        self.parser.add_argument(
            '--embedding_layers',
            type=int,
            default=1,
        )
        self.parser.add_argument(
            '--embedding_size',
            type=int,
            default=20,
        )
        self.parser.add_argument(
            '--dc_loss_ratio',
            type=float,
            default=0.5,
        )
        self.parser.add_argument(
            '--logit_threshold',
            type=float,
            default=0.5,
        )
        self.parser.add_argument(
            '--early_stopping', 
            type = int, 
            default = 25,
            help = "Training is stopped if there are early_stopping consectuive validations without improvement. \
                Set to 0 if you don't want to execute this utility.",
        )
        self.parser.add_argument(
            '--use_weights_and_biases',
            action = argparse.BooleanOptionalAction,
            default = True,
            help = 'Set to True if you want to use Weights and Biases.',
            )
        self.parser.add_argument(
            '--wandb_dir',
            type = str, 
            default= "/home/bsc/bsc088135/Diarization/wandb",
            help = 'An absolute path to the directory where Weight & Biases metadata and downloaded files will be stored, \
            when use_weights_and_biases is True. If not specified, this defaults to the ./wandb directory.',
            )

        return self.parser.parse_args()
    
