from torch.utils import data
import pandas as pd
from typing import Tuple
import os
from io import StringIO

class TrainDataset(data.Dataset):
    def __init__(self, parameters):
        self.audio_path = parameters.audio_path
        self.transcriptions_path = parameters.transcriptions_path
        self.open_files()

    def read_tdf(self, path) -> Tuple[dict, pd.DataFrame]:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        metadata_lines = []
        data_lines = []

        for line in lines:
            if line.startswith(';;'):
                metadata_lines.append(line)
            else:
                data_lines.append(line)
        print(f"metadata_lines: {metadata_lines}")
        print(f"data_lines: {data_lines}")
        # Extract metadata.
        metadata = {}
        for meta_line in metadata_lines:
            if "sectionTypes" in meta_line:
                metadata["sectionTypes"] = meta_line.split()[-1]  # Assuming it's the last element
            elif "sectionBoundaries" in meta_line:
                metadata["sectionBoundaries"] = meta_line.split()[-1]
        # Define column names based on the file's header structure
        column_names = [
            "file", "channel", "start", "end", "speaker", "speakerType",
            "speakerDialect", "transcript", "section", "turn", "segment"
        ]
        # Read the data part into a DataFrame

        data = pd.read_csv(
            StringIO("\n".join(data_lines)),
            delim_whitespace=True,
            names=column_names
        )

        return metadata, data        

    def open_files(self):
        for root, paths, files in os.walk(self.transcriptions_path):
            for file in files:
                print(f"path: {os.path.join(root,file)}")
                metadata, data = self.read_tdf(os.path.join(root,file))
            print(f"metadata: {metadata}")
            print(f"data: {data}")
    
    def __len__(self):
        ...
    def __getitem__(self, index):
        ...