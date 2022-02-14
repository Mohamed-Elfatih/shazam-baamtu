'''
this module will:
1 - downsample data to specified frequency
2-  generate test data
'''

import os
from .utils import RIRNoise, BackgroundNoise, clip_audio, _list_all_filepaths, AudioBackend
import torchaudio
import glob
from tqdm import tqdm
from termcolor import colored
import librosa
import numpy as np
import torch
import argparse
import shutil

# TODO: consisteny in names
# TODO: documentation
# TODO: type checking
# TODO: arguments
# TODO: delete folder if they exists

# parent directory where audios can be found
parent_dir = '/Users/mohamedelfatih/Documents/baamtu/project/simquran-machine-learning'
# path to the audios 
AUDIO_DIR = os.path.join(parent_dir, './data_processed/test_chunk')
# path to the downloaded noises
BACKGROUND_NOISE_DIR = os.path.join(parent_dir, './picked_noise/Background')
RIR_NOISE_DIR = os.path.join(parent_dir, './picked_noise/RIR')
# path to the output folder
OUTPUT_TEST_DIR = './augmented_audios'
OUTPUT_DOWNSAMPLED_DIR = './downsampled_audios'


def downsample_audios(audio_dir, frequency, audio_backend, out_dir):
    filepaths = _list_all_filepaths(audio_dir)
    print(colored("Start to downsample audios", "green"))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for filepath in tqdm(filepaths):
        # read the audio
        waveform, fs = audio_backend.load(filepath)
        # downsample
        waveform = librosa.resample(np.array(waveform, dtype=np.float32), fs, frequency, res_type='kaiser_best')
        fs = frequency
        waveform = torch.tensor(waveform).float()
        # save
        filename = filepath.split('/')[-1][:-4]
        audio_backend.save(os.path.join(out_dir, filename + '.mp3'), waveform, fs)


 
def generate_aug_audios(noise_dir, audio_dir, out_dir, noise_type, interval, audio_backend, db = 10):
    '''
    @author: MohamedElfatih

    this function generates augmentation, it supports two types which 
    are "add background noies" or "RIR"

    Parameters
    -----------------------------------------------------
    noise_dir: str, the path to noise audios
    audio_dir: str, the path to audios to be augmented
    out_dir: str, the path to directory to save results
    noise_type: str, either "background" or "rir"
    frequency: int, the frequency of the saved audios,
    interval: int, number of the seconds to be clipped
    db: int, the ratio used for background noise

    Returns
    -----------------------------------------------------
    None
    '''
    # TODO: Type checking 
    #
    noises_filepaths = _list_all_filepaths(noise_dir)
    filepaths = _list_all_filepaths(audio_dir)
    for noise_filepath in noises_filepaths:
        noise_name = noise_filepath.split('/')[-1][:-4]
        noise_type_dir  = os.path.join(out_dir, noise_name)
        if not os.path.exists(noise_type_dir):
            os.mkdir(noise_type_dir)
        
        if noise_type == 'rir':
            noise= RIRNoise(
                dir=None,
                uri=None,
                filename=None,
                local=True,
                filepath=noise_filepath,
            )
        elif noise_type == 'background':
            noise= BackgroundNoise(
                dir=None,
                uri=None,
                filename=None,
                local=True,
                filepath=noise_filepath,
                db = db
            )
        
        for filepath in tqdm(filepaths):
            filename = filepath.split('/')[-1][:-4] + '.mp3'
            waveform, sample_rate = audio_backend.load(filepath)
            waveform = clip_audio(waveform, sample_rate, interval)
            new_sample_rate = sample_rate 
            if noise != None:
                waveform, new_sample_rate = noise.apply(waveform, sample_rate)

            # if new_sample_rate > frequency:
            #     if waveform.shape[0] == 1:
            #         waveform = waveform.repeat(2,1)
            #     waveform = np.array(waveform, dtype=np.float32)
            #     #waveform = librosa.resample(waveform, new_sample_rate, frequency, res_type='kaiser_best')
            #     #new_sample_rate = frequency
            #     waveform = torch.tensor(waveform).float()
            
            audio_backend.save(os.path.join(noise_type_dir, filename), waveform, new_sample_rate)

def generate_clean_date(audio_dir, out_dir, frequency, interval, audio_backend):
    '''
    Parameters
    -----------------------------------------------------
    audio_dir: str, path for the audio directory,
    out_dir: str, the path to the output directory
             where to save generate audios
    frequency: int, the downsampling frequency
    interval: int, the interval in seconds to clip from audio
    audio_backend: AudioBackend, a module for I/O operations (load and save)

    Returns
    -----------------------------------------------------
    None
    '''
    clean_dir = os.path.join(out_dir, './clean')
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
    filepaths = _list_all_filepaths(audio_dir)
    for filepath in tqdm(filepaths):
            filename = filepath.split('/')[-1]
            waveform, sample_rate = audio_backend.load(filepath)
            waveform = clip_audio(waveform, sample_rate, interval)
            # waveform = librosa.resample(np.array(waveform, dtype=np.float32), sample_rate, frequency, res_type='kaiser_best')
            # waveform = torch.tensor(waveform).float()
            filepath = os.path.join(clean_dir, filename)
            audio_backend.save(filepath, waveform, sample_rate)
            
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='process the downsampling frequency, interval and snr ratio')
    parser.add_argument('--db', default=10, type=int, help='SNR ratio')
    parser.add_argument('--t', default=6, type=int, help='interval to clip')
    parser.add_argument('--f', default=11500, type=int, help='downsampling frequency')
    parser.add_argument('--downsample', default=False, type=bool, help='whether to create a downsample audios from which we create the fingerprints')


    args = parser.parse_args()
    frequency= args.f
    interval = args.t
    db = args.db
    is_downsample = args.downsample

    audio_backend = AudioBackend()

    if is_downsample:
        downsample_audios(AUDIO_DIR, frequency, audio_backend, OUTPUT_DOWNSAMPLED_DIR)

    # if the augmented folder exist delete it
    if os.path.exists(OUTPUT_TEST_DIR):
        print("clearing the augmented audios folder")
        shutil.rmtree(OUTPUT_TEST_DIR)
        print("creating an empty augmented audios folder")
        os.mkdir(OUTPUT_TEST_DIR)
    
    # background
    print(colored("started creating background version", 'yellow'))
    generate_aug_audios(BACKGROUND_NOISE_DIR, OUTPUT_DOWNSAMPLED_DIR, OUTPUT_TEST_DIR, 'background', interval, audio_backend=audio_backend, db=db)
    print(colored("finished the background version", 'green'))

    # rir 
    print(colored("started creating rir version", 'yellow'))
    generate_aug_audios(RIR_NOISE_DIR, OUTPUT_DOWNSAMPLED_DIR, OUTPUT_TEST_DIR, 'rir', interval, audio_backend=audio_backend)
    print(colored("finished the rir version", 'green'))

    # clean
    print(colored("start creating the clean version", "yellow"))
    generate_clean_date(OUTPUT_DOWNSAMPLED_DIR, OUTPUT_TEST_DIR, frequency, interval, audio_backend)
    print(colored("finished the clean version", 'green'))
    
    

     
    
    

   
   

   
    





    
