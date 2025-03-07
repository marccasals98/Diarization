#!/usr/bin/env python
import os
import glob
import argparse
import numpy as np

def read_rttm(rttm_path: str) -> list:
    """
    Reads an RTTM file and returns a list of tuples:
    (start_time, duration, speaker)
    """
    turns = []
    with open(rttm_path, 'r') as f:
        for line in f:
            # Skip comments or empty lines
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                turns.append((start, duration, speaker))
            except ValueError:
                continue
    return turns

def compute_overlap(turns: list) -> tuple:
    """
    Given a list of speaker turns (start, duration, speaker), computes:
      - max_simultaneous: maximum number of speakers active at any time
      - overlap_time: total time during which more than one speaker is active
      - total_duration: total duration covered by the RTTM file
      - overlap_ratio: fraction of total time that has overlapping speech
    Uses a sweep-line algorithm over time.
    """
    events = []
    for start, duration, _ in turns:
        end = start + duration
        events.append((start, 1))
        events.append((end, -1))
    if not events:
        return 0, 0.0, 0.0, 0.0

    # Sort events by time; in case of ties, end events (-1) come before start events (+1)
    events.sort(key=lambda x: (x[0], x[1]))
    
    max_simultaneous = 0
    current_active = 0
    overlap_time = 0.0
    last_time = events[0][0]
    
    for time, delta in events:
        if time > last_time:
            interval = time - last_time
            if current_active > 1:
                overlap_time += interval
            last_time = time
        current_active += delta
        max_simultaneous = max(max_simultaneous, current_active)
    
    # Total duration is from the earliest start to the latest end
    total_duration = events[-1][0] - events[0][0]
    overlap_ratio = overlap_time / total_duration if total_duration > 0 else 0.0
    
    return max_simultaneous, overlap_time, total_duration, overlap_ratio

def process_rttm_file(rttm_path: str) -> dict:
    """
    Process a single RTTM file and returns statistics as a dictionary.
    """
    turns = read_rttm(rttm_path)
    # Unique speakers in the file:
    unique_speakers = {speaker for _, _, speaker in turns}
    num_unique = len(unique_speakers)
    
    max_simul, overlap_time, total_duration, overlap_ratio = compute_overlap(turns)
    
    return {
        'file': os.path.basename(rttm_path),
        'num_unique_speakers': num_unique,
        'max_simultaneous': max_simul,
        'total_duration': total_duration,
        'overlap_time': overlap_time,
        'overlap_ratio': overlap_ratio,
    }

def main():
    parser = argparse.ArgumentParser(description="Compute RTTM file statistics.")
    parser.add_argument("rttm_folder", type=str, help="Path to folder containing RTTM files.")
    args = parser.parse_args()

    rttm_folder = args.rttm_folder
    rttm_files = glob.glob(os.path.join(rttm_folder, "*.rttm"))
    if not rttm_files:
        print("No RTTM files found in the provided folder.")
        return

    all_stats = []
    print("Per-file statistics:")
    for file_path in rttm_files:
        stats = process_rttm_file(file_path)
        all_stats.append(stats)
        print(f"File: {stats['file']}")
        print(f"  Unique speakers: {stats['num_unique_speakers']}")
        print(f"  Max simultaneous speakers: {stats['max_simultaneous']}")
        print(f"  Total duration: {stats['total_duration']:.2f} sec")
        print(f"  Overlap time: {stats['overlap_time']:.2f} sec")
        print(f"  Overlap ratio: {stats['overlap_ratio']:.2f}")
        print("")

    # Compute overall aggregated statistics
    avg_unique = np.mean([s['num_unique_speakers'] for s in all_stats])
    max_unique = np.max([s['num_unique_speakers'] for s in all_stats])
    avg_max_simul = np.mean([s['max_simultaneous'] for s in all_stats])
    overall_overlap_ratio = np.mean([s['overlap_ratio'] for s in all_stats])

    print("Overall statistics:")
    print(f"  Average unique speakers per file: {avg_unique:.2f}")
    print(f"  Maximum unique speakers in any file: {max_unique}")
    print(f"  Average max simultaneous speakers: {avg_max_simul:.2f}")
    print(f"  Overall average overlap ratio: {overall_overlap_ratio:.2f}")

if __name__ == "__main__":
    main()
