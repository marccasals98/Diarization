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
    ref_mat = np.atleast_2d(np.array(reference_matrix, dtype=int))
    hyp_mat = np.atleast_2d(np.array(hypothesis_matrix, dtype=int))

    # Ensure that the arrays are at least 2D
    if ref_mat.ndim == 1:
        # Assuming that a 1D array means (num_speakers,) with one time frame
        ref_mat = ref_mat.reshape(-1, 1)
    if hyp_mat.ndim == 1:
        hyp_mat = hyp_mat.reshape(-1, 1)

    ref_mat = ref_mat.transpose()
    hyp_mat = hyp_mat.transpose()

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
                end_time = (
                    prev_frame + 1
                ) * frame_duration  # end of segment is exclusive of this frame
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


def compute_der_batch(reference_batch, hypothesis_batch, frame_duration):
    """
    Compute DER for a batch of diarization examples.

    Parameters:
        reference_batch (array-like): Binary matrices with shape
                                      (batch_size, num_ref_speakers, num_frames).
        hypothesis_batch (array-like): Binary matrices with shape
                                       (batch_size, num_hyp_speakers, num_frames).
        frame_duration (float): Duration (in seconds) of each time frame.

    Returns:
        list: DER values for each sample in the batch.
              You can also compute an average if desired.
    """
    reference_batch = np.array(reference_batch)
    hypothesis_batch = np.array(hypothesis_batch)

    if reference_batch.shape[0] != hypothesis_batch.shape[0]:
        raise ValueError("Batch size mismatch between reference and hypothesis")

    ders = []
    for ref_mat, hyp_mat in zip(reference_batch, hypothesis_batch):
        der = compute_der(ref_mat, hyp_mat, frame_duration)
        ders.append(der)

    # Optionally, compute the average DER over the batch:
    average_der = np.mean(ders)
    return ders, average_der
