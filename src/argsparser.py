import argparse

class ArgsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Speaker Diarization test.')
    
    def parse_args(self):
        self.parser.add_argument(
            '--audio_path',
            type=str,
            default='/gpfs/projects/bsc88/speech/data/raw_data/diarization/fisher_spa/data/speech',
        )
        self.parser.add_argument(
            '--transcriptions_path',
            type=str,
            default='/gpfs/projects/bsc88/speech/data/raw_data/diarization/fisher_spa_tr/data/transcripts',
        )
        self.parser.add_argument(
            '--max_epochs',
            type=int,
            default=5,
        )
    
        return self.parser.parse_args()
    
