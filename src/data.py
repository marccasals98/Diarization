from torch.utils import data
import pandas as pd
from typing import Tuple
import os
from io import StringIO
import torchaudio
import numpy as np
import torch
from tqdm import tqdm
import ipdb

def load_audio(audio_path: str)->list:
    audio, sr = torchaudio.load(audio_path)
    return audio, sr

def read_rttm(rttm_path: str)->Tuple[list, list]:
    """This function reads rttm files and returns the speaker turns and a list of the speakers.

    We want a list of the speakers because it is the most optimal way to calculate
    the dimensions of the label vector.

    Args:
        rttm_path (str): The path to the rttm file.

    Returns:
        Tuple[list, list]: The speaker turns and the speakers.
    """
    results = []
    speakers = []
    with open(rttm_path, 'r') as f:
        for line in f:
            # Skip comment lines and any blank lines
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            # Make sure the line has at least 8 elements
            if len(parts) >= 8:
                # Extract the 4th, 5th, and 8th elements.
                # (Indexes 3, 4, and 7 in Python's 0-indexed lists)
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                results.append((start, duration, speaker))
                if speaker not in speakers:
                    speakers.append(speaker)
                else:
                    continue
    return results, speakers

def get_files_of_path(path: str)->list:
    """Given a path, this function returns a list of all the files in that path.

    Args:
        path (str): The path we want to retrieve the files.

    Returns:
        list: The list with the correspondant files.
    """
    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def speaker_to_index(speaker_list: list)->dict:
    """Maps speaker names to indices.

    {"speakerA": 0, "speakerB": 1, "speakerC": 2}

    Args:
        speaker_list (list): the list of speaker names

    Returns:
        dict: mapping of speaker names to indices
    """
    speaker_indices = {}
    for i, speaker in enumerate(speaker_list):
        speaker_indices[speaker] = i
    return speaker_indices

def pad_labels(labels: np.array, target_num: int) -> np.array:
    """
    Pads the label matrix to have a fixed number of columns (speakers).
    
    Args:
        labels (np.array): Label matrix of shape (num_frames, current_num_speakers)
        target_num (int): The desired number of speakers (columns)
    
    Returns:
        np.array: Label matrix padded to shape (num_frames, target_num)
    """
    num_frames, current_num = labels.shape
    if current_num < target_num:
        pad_width = ((0, 0), (0, target_num - current_num))
        labels = np.pad(labels, pad_width, mode='constant')
    return labels

class TrainDataset(data.Dataset):
    def __init__(self, 
                audio_files: str,
                rttm_paths: str,
                feature_stride: float, 
                n_frames: int,
                segment_length: int, 
                allow_overlap: bool, 
                max_num_speakers: int,
                transform=None,
                frame_length=0.025):
        
        self.audio_files = audio_files
        self.rttm_paths = rttm_paths
        self.feature_stride = feature_stride
        self.n_frames = n_frames
        self.segment_length = segment_length
        self.allow_overlap = allow_overlap
        self.max_num_speakers = max_num_speakers # HACK implement maximum number of speakers
        self.transform = transform
        self.frame_length = frame_length # in seconds
        self.precompute_segments()

    def precompute_segments(self):
        """This function reads the audio files and RTTM files from the paths
        provided in the constructor and precomputes the segments for the
        dataset.
        """
        print("Precomputing segments...")
        self.segments = []
        audio_files = sorted(get_files_of_path(self.audio_files))
        rttm_files = sorted(get_files_of_path(self.rttm_paths))

        if len(audio_files) != len(rttm_files):
            print(f"audio_files: {len(audio_files)}")
            print(f"rttm_files: {len(rttm_files)}")
            raise ValueError("The number of audio files and RTTM files must match.")

        for audio_file, rttm_file in tqdm(zip(audio_files, rttm_files), total=len(audio_files)):
            turns, speakers = read_rttm(rttm_file)
            audio, sr = load_audio(audio_file)
            n_samples =len(audio[0])
            seg_samples = int(self.segment_length * sr)

            # we iterate over the start of each segment
            for start in range(0, n_samples-seg_samples + 1, seg_samples):
                labels = self.compute_labels_for_segment(turns, start / sr, (start + seg_samples) / sr, speakers)
                self.segments.append((audio_file, start / sr, labels))

    def compute_labels_for_segment(self, turns: list, segment_start: float, segment_end: float, speakers: list)->np.array:
        """This function converts the speaker turn information from your RTTM
        file into a frame-level label vector for a fixed-length audio segment.

        Args:
            turns (list): the list of speaker turns
            start (float): the start of the segment
            end (float): the end of the segment

        Returns:
            np.array: the frame-level label vector that comprehends the speaker turns
        """


        # Create a mapping of speaker names to indices
        num_speakers = len(speakers)
        speaker_indices = speaker_to_index(speakers)

        # Initialize the label vector
        if self.allow_overlap == True:
            label_matrix = np.zeros((self.n_frames, num_speakers), dtype=int)
        else:
            label_matrix = np.zeros(self.n_frames, dtype=int)

        # Iterate over the speaker turns
        for start_time, duration, speaker in turns:
            end_time_turn = start_time + duration
            # We need to check if the speaker turn is within the segment
            if end_time_turn > segment_start and start_time < segment_end:

                # Convert the start and end times to frame indices
                # Since feature_stride is in seconds, the number of frames from segment_start is:
                start_frame = max(0, int((start_time - segment_start) / self.feature_stride))
                end_frame = min(self.n_frames, int((end_time_turn - segment_start) / self.feature_stride))

                # Set the label vector values for the speaker turn
                if self.allow_overlap == True:
                    label_matrix[start_frame:end_frame, speaker_indices[speaker]] = 1
                else:
                    label_matrix[start_frame:end_frame] = speaker

        return label_matrix
    
    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index: int)->Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        The audio returned is of length `segment_length` seconds. In number of samples it is `segment_length * sr`.

        The labels are a binary matrix of shape `(num_frames, num_speakers)`.
        Each row corresponds to a frame in the audio segment, and each column corresponds to a speaker. 
        If a speaker is speaking in a frame, the corresponding column will be 1, otherwise 0.

        Args:
            index (int): The index of the sample we want to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns an audio of shape `segment_length * sr` and the labels of shape `(num_frames, num_speakers)`. Another way `(segment_length/frame_length, num_speakers)`.
            Usually, the code is set to work with segment_length = 5, and Sampling rate = 16000.
        """
        audio_file, start_time, labels = self.segments[index]
        audio, sr = load_audio(audio_file)
        start_sample = int(start_time * sr)
        seg_samples = int(self.segment_length * sr)
        audio_segment = audio[:, start_sample:start_sample + seg_samples]

        # pad the audio segment if it's shorter than the segment length
        if len(audio_segment[0]) < seg_samples:
            pad_length = seg_samples - len(audio_segment[0])
            audio_segment = np.pad(audio_segment, (0, pad_length), mode='constant')
        if self.transform:
            audio_segment = self.transform(audio_segment)
        
        # If required, pad the labels to match the maximum number of speakers.
        if self.max_num_speakers is not None:
            labels = pad_labels(labels, self.max_num_speakers)
        # Convert to tensors:
        audio_tensor = audio_segment.clone().detach().float()
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return audio_tensor, label_tensor
