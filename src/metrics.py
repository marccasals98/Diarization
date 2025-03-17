import numpy as np
from scipy.optimize import linear_sum_assignment
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

def compute_der(reference_matrix, hypothesis_matrix, frame_duration):
    """
    Compute the Diarization Error Rate (DER) between a reference and hypothesis diarization.
    
    Parameters:
        reference_matrix (array-like): Binary matrix of shape (N_ref_speakers, T) for ground truth.
        hypothesis_matrix (array-like): Binary matrix of shape (N_hyp_speakers, T) for prediction.
        frame_duration (float): Duration of each time frame (in seconds).
    
    Returns:
        float: Diarization Error Rate (fraction of time that is diarization error).
    """
    # Convert inputs to numpy arrays
    ref_mat = np.array(reference_matrix, dtype=int)
    hyp_mat = np.array(hypothesis_matrix, dtype=int)
    ref_n_speakers, ref_time = ref_mat.shape
    hyp_n_speakers, hyp_time = hyp_mat.shape

    # Check that the number of frames is the same for reference and hypothesis
    if ref_time != hyp_time:
        raise ValueError("Reference and hypothesis must have the same number of frames")
    
    # Compute overlap (in frames) between each reference speaker i and predicted speaker j
    # overlap[i, j] = number of frames where ref_speaker i and hyp_speaker j are both active
    overlap = ref_mat.dot(hyp_mat.T)  # shape (G, P)
    
    # Find optimal one-to-one mapping between reference and hypothesis speakers to maximize overlap
    # (Hungarian algorithm solves minimum cost, so we negate overlap or use maximize=True if available)
    row_ind, col_ind = linear_sum_assignment(overlap, maximize=True)
    # row_ind: indices of reference speakers, col_ind: indices of corresponding mapped hypothesis speakers
    
    # Build reference annotation with segments for each true speaker
    reference = Annotation()
    for speaker_ref in range(ref_n_speakers):
        speaker_label = f"Speaker{speaker_ref}"
        # Find frames where speaker i is active
        frames = np.where(ref_mat[speaker_ref] == 1)[0]
        if frames.size == 0:
            continue  # no speech for this speaker
        # Group consecutive frames into segments
        start_frame = frames[0]
        prev_frame = frames[0]
        for frame in frames[1:]:
            if frame != prev_frame + 1:
                # End of a continuous segment
                start_time = start_frame * frame_duration
                end_time = (prev_frame + 1) * frame_duration  # end of segment is exclusive of this frame
                reference[Segment(start_time, end_time)] = speaker_label
                start_frame = frame  # start new segment
            prev_frame = frame
        # Add last segment for this speaker
        start_time = start_frame * frame_duration
        end_time = (prev_frame + 1) * frame_duration
        reference[Segment(start_time, end_time)] = speaker_label
    
    # Build hypothesis annotation with segments for each predicted speaker (using mapped labels)
    hypothesis = Annotation()
    # Create a mapping from predicted speaker index -> reference speaker label (if assigned)
    mapping = {pj: f"Speaker{ri}" for ri, pj in zip(row_ind, col_ind)}
    for speaker_hyp in range(hyp_n_speakers):
        # Determine label: use mapped reference label if available, else a unique hypothesis label
        hyp_label = mapping.get(speaker_hyp, f"Speaker_hyp{speaker_hyp}")
        frames = np.where(hyp_mat[speaker_hyp] == 1)[0]
        if frames.size == 0:
            continue
        # Group consecutive frames into segments
        start_frame = frames[0]
        prev_frame = frames[0]
        for frame in frames[1:]:
            if frame != prev_frame + 1:
                # end current segment
                start_time = start_frame * frame_duration
                end_time = (prev_frame + 1) * frame_duration
                hypothesis[Segment(start_time, end_time)] = hyp_label
                start_frame = frame
            prev_frame = frame
        # Add last segment for this predicted speaker
        start_time = start_frame * frame_duration
        end_time = (prev_frame + 1) * frame_duration
        hypothesis[Segment(start_time, end_time)] = hyp_label
    
    # Compute DER using pyannote.metrics
    metric = DiarizationErrorRate() 
    der_value = metric(reference, hypothesis)
    return der_value

