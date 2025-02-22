# Diarization
Speaker Diarization is the realm of techniques that focus on answering "Who spoke when?". The main goal of this repo is to test state-of-the-art diarization systems. To do so, we will test different speaker diarization datasets. The initial efforts will consider the Fisher Spanish Speech, found in the [Linguistic Data Consortium](	https://doi.org/10.35111/skrw-t863).

## 0. Converting files to RTTM.
There exist multiple formats from which speaker diarization is represented:
Rich Transcription Time Marked (RTTM) files are space-delimited text files containing one turn per line, each line containing ten fields:

* Type -- segment type; should always by SPEAKER
* File ID -- file name; basename of the recording minus extension (e.g., rec1_a)
* Channel ID -- channel (1-indexed) that turn is on; should always be 1
* Turn Onset -- onset of turn in seconds from beginning of recording
* Turn Duration -- duration of turn in seconds
* Orthography Field -- should always by < NA >
* Speaker Type -- should always be < NA >
* Speaker Name -- name of speaker of turn; should be unique within scope of each file
* Confidence Score -- system confidence (probability) that information is correct; should always be < NA >
* Signal Lookahead Time -- should always be < NA >

In the Fisher Spanish Speech, there appear 819 `.tdf` files. We will convert them to `.rttm` by using `utils/convert_to_rttm.py`.
