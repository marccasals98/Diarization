# MIT License
#
# Marc Casals i Salvador
# @BSC-CNS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""This software is intended to convert .tdf files into rttm files"""

import pandas as pd
import argparse
import os
from tqdm import tqdm

class ArgsParser:
    """
    This class is createed for parsing the arguments of the script.
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='Speaker Diarization test.')
    
    def parse_args(self):   
        self.parser.add_argument(
            '--tdf_folder',
            type=str,
            default='/gpfs/projects/bsc88/speech/data/raw_data/diarization/fisher_spa_tr/data/transcripts',
        )
        self.parser.add_argument(
            '--output_folder',
            type=str,
            default='/gpfs/projects/bsc88/speech/data/raw_data/diarization/fisher_spa_tr/data/rttm_transcripts',
        )
        return self.parser.parse_args()

def tdf_files(tdf_folder: str)->list:
    """This function reads all the `.tdf` files from a folder and returns a list with the paths.

    Args:
        tdf_folder (str): The folder where the `.tdf` files are.

    Returns:
        list: A list with the paths of the `.tdf` files.
    """
    tdf_files = []
    for folder, _, files in os.walk(tdf_folder):
        for file in files:
            if file.endswith('.tdf'):
                tdf_files.append(os.path.join(folder, file))
    return tdf_files

def format_dataframe(df: pd.DataFrame)->pd.DataFrame:
    """This function formats the dataframe to the format of the rttm file.

    Args:
        df (pd.DataFrame): The dataframe that we want to format.

    Returns:
        pd.DataFrame: The formatted dataframe.
    """
    # Create all the necessary columns for rttm format.
    df['type'] = 'SPEAKER'
    df['duration'] = df['end'] - df['start']
    df['ortho'] = '<NA>'
    df['stype'] = '<NA>'
    df['conf'] = '<NA>'
    df['slat'] = '<NA>'
    
    # Change the ordering and discarding non-rttm columns.
    df = df[["type", "file", "channel", "start", "duration", "ortho", "stype", "speaker", "conf", "slat"]]

    return df

def main():

    args = ArgsParser().parse_args()

    # We create a list with all the `.tdf` files in the folder
    tdf_files_list = tdf_files(args.tdf_folder)
    for tdf_file in tqdm(tdf_files_list):
        df = pd.read_csv(tdf_file, sep='\t', header=None, skiprows=[0,1,2])
        # We rename the columns according to Fisher Spanish dataset:
        df.columns = ["file", "channel", "start", "end",
                    "speaker", "speakerType", "speakerDialect","transcript",
                    "section", "turn", "segment", "sectionTyp", "suType"]
        
        # We format the dataframe into the rttm.
        df = format_dataframe(df)
        rttm_file = os.path.join(args.output_folder, os.path.basename(tdf_file).replace('.tdf', '.rttm'))
        df.to_csv(rttm_file, sep=' ', index=False, header=False)

if __name__ == '__main__':
    main()