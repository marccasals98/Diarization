# Diarization
The main goal of this repo is to test state of the art diarization systems. 

## 0. Converting files to RTTM.
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

We have 819 `.tdf` files that contain all this information. We need to convert these files into 3 different files: 
1. `train.rttm`
2. `dev.rttm`
3. `test.rttm`
For doing so, we are going to open all these files with Pandas and concatenate them, creating a big dataframe with all of the instances. Then doing this partition into different files.