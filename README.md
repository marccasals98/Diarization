# Diarization
Speaker Diarization is the realm of techniques that focus on answering "Who spoke when?". The main goal of this repo is to test state-of-the-art diarization systems. To do so, we will test different speaker diarization datasets. The initial efforts will consider the [VoxConverse](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/) dataset.

## 0. Converting files to RTTM.
There exist multiple formats in which speaker diarization is represented:
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

VoxConverse is an audio-visual diarization dataset consisting of over 50 hours of multispeaker clips of human speech extracted from YouTube videos. 

## 1. Loading Data.
This step entirely depends on the model we want to use. In this code, we start using the EEND approach discussed in the paper "End-to-End Neural Speaker Diarization with Permutation-Free Objectives" from Fujita et al. They substantially address Speaker Diarization as a multiclass classification problem. For this reason, an observation sequence $X = \left(x_t\in\mathbb{R}^F \mid t =1,...,T\right)$ must be frame-to-frame aligned with the speaker label sequence $Y=(y_t \mid t=1,...,T)$.

The initial feature extractor that we tried was the Mel Spectrogram. However, we may want to use other SSL approaches in the future.

## Bibliography
J. S. Chung*, J. Huh*, A. Nagrani*, T. Afouras, A. Zisserman
Spot the conversation: speaker diarisation in the wild  
Interspeech, 2020

Fujita Y, Kanda N, Horiguchi S, Nagamatsu K, Watanabe S. End-to-End Neural Speaker Diarization with Permutation-Free Objectives. In: Interspeech 2019 [Internet]. ISCA; 2019 [cited 2024 Oct 26]. p. 4300–4. Available from: https://www.isca-archive.org/interspeech_2019/fujita19_interspeech.html
