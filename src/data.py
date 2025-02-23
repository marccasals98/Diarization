from torch.utils import data
import pandas as pd
from typing import Tuple
import os
from io import StringIO
import torchaudio
import numpy as np
import torch

def load_audio(audio_path)->list:
    audio, sr = torchaudio.load(audio_path)
    return audio, sr

def read_rttm(rttm_path)->list:
    with open(rttm_path, 'r') as f:
        lines = f.readlines()
    list_lines = []
    for line in lines:
        list_lines.append(line.split())
    return list_lines

class TrainDataset(data.Dataset):
    def __init__(self, 
                audio_path,
                transcriptions_path, 
                segment_length, 
                allow_overlap, 
                num_speakers,
                frame_length=100):
        
        self.audio_path = audio_path
        self.transcriptions_path = transcriptions_path
        self.segment_length = segment_length
        self.allow_overlap = allow_overlap
        self.num_speakers = num_speakers
        self.frame_length = frame_length
        self.open_files()

    def precompute_segments(self):
        self.segments = []
        for audio_file, rttm_file in zip(self.audio_files, self.rttm_paths):
            turns = read_rttm(rttm_file)
            audio, sr = load_audio(audio_file)
            n_samples =len(audio[0])
            seg_samples = int(self.segment_length * sr)

            for start in range(0, n_samples-seg_samples + 1, seg_samples):
                labels = self.compute_labels_for_segment(turns, start / sr, (start + seg_samples) / sr)
                self.segments.append((audio_file, start / sr, labels))

    def compute_labels_for_segment(self, turns: list, start: float, end: float)->np.array:
        """This function converts the speaker turn information from your RTTM
        file into a frame-level label vector for a fixed-length audio segment.

        Args:
            turns (list): _description_
            start (float): _description_
            end (float): _description_

        Returns:
            np.array: _description_
        """
        # HACK put frames per second as hyperparemeter.
        n_frames = int(self.segment_length * self.frame_length) # self.frame_length frames per second
        if self.allow_overlap == True:
            label_vector = np.zeros((n_frames, self.num_speakers), dtype=int)
        else:
            label_vector = np.zeros(n_frames, dtype=int)
        
        for start_time, duration, speaker in turns:
            end_time = start_time + duration
            # We need to check if the speaker turn is within the segment
            if end_time > start and start_time < end:
                start_frame = max(0, int((start_time - start) * self.frame_length))
                end_frame = min(n_frames, int((end_time - start) * self.frame_length))

                if self.allow_overlap == True:
                    label_vector[start_frame:end_frame, speaker] = 1
                else:
                    label_vector[start_frame:end_frame] = speaker

        return label_vector
    
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
        return len(self.segments)
    
    def __getitem__(self, index):
        audio_file, start_time, labels = self.segments[index]
        audio, sr = load_audio(audio_file, self.sr)
        start_sample = int(start_time * sr)
        seg_samples = int(self.segment_length * sr)
        audio_segment = audio[:, start_sample:start_sample + seg_samples]

        # pad the audio segment if it's shorter than the segment length
        if len(audio_segment[0]) < seg_samples:
            pad_length = seg_samples - len(audio_segment[0])
            audio_segment = np.pad(audio_segment, (0, pad_length), mode='constant')
        if self.transform:
            audio_segment = self.transform(audio_segment)
        
        # Convert to tensors:
        audio_tensor = torch.tensor(audio_segment, dtype=torch.float)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return audio_tensor, label_tensor