import os
import io
import tarfile
import math
import torch
import torchaudio
import requests
import librosa
import numpy as np
from abc import ABC, abstractmethod
from termcolor import colored
from timeit import timeit
from itertools import zip_longest
from dejavu import Dejavu
from dejavu.logic.recognizer.file_recognizer import FileRecognizer
from dejavu import Dejavu

config = {
      "database": {
          "host": "127.0.0.1",
          "user": "root",
          "password": '2541997m',
          "database": 'dejavu'}
        }
djv = Dejavu(config)
# ================================== NOISE ================================== #
class Noise(ABC):
    def __init__(self, dir, uri, filename, local=False, filepath=None):
        self.dir = dir
        self.uri = uri
        self.filename = filename
        if not local:
            self.fetch_data()
        else:
            assert filepath is not None, 'you should pass a filepath'
            self.file_path = filepath
    
    def fetch_data(self):
        '''
        fetch noise data from uri and save 
        it in dir
        '''
        assert self.uri is not None, 'uri is None'
        assert self.dir is not None, 'dir is None'
        #assert os.path.exists(dir), 'dir doesn\'t exist'
        assert self.filename is not None, 'filname is None'

        self.file_path = os.path.join(self.dir, self.filename)
        with open(self.file_path, 'wb') as file_:
            file_.write(requests.get(self.uri).content)
    
    def get_sample(self, rsample=None):
        path =  self.file_path
        effects = [["remix", "1"]]
        if rsample:
            effects.extend(
                [
                    ["lowpass", f"{rsample // 2}"],
                    ["rate", f"{rsample}"],
                ]
            )
        return torchaudio.sox_effects.apply_effects_file(path, effects=effects)
    
    def downsample(self, waveform, sample_rate, noise_sample_rate):
        new_sample_rate = sample_rate
        if noise_sample_rate < sample_rate:
            new_sample_rate = noise_sample_rate
            waveform = torchaudio.functional.resample(waveform, sample_rate, noise_sample_rate)
        return new_sample_rate, waveform

    @abstractmethod
    def apply(self, waveform, sample_rate:int):
        pass

class RIRNoise(Noise):
    def __init__(self, dir, uri, filename, local, filepath, rir_sample_rate=20):
        super().__init__(dir, uri, filename, local, filepath)
        self.rir_sample_rate = rir_sample_rate
    

    def apply(self, waveform, sample_rate):
        rir_raw, noise_sample_rate = self.get_sample()
        new_sample_rate, waveform = self.downsample(waveform, sample_rate, noise_sample_rate)
        
        start = int(self.rir_sample_rate * 1.01)
        end = int(self.rir_sample_rate * 1.3)
       
        rir = rir_raw[:, start:end]
        rir = rir / torch.norm(rir, p=2)
        rir = torch.flip(rir, [1])

        num_of_channels = len(waveform[:])
        rir = rir.repeat(num_of_channels, 1)

        waveform_ = torch.nn.functional.pad(waveform, (rir.shape[1] - 1, 0))
        augmented = torch.nn.functional.conv1d(waveform_[None, ...], rir[None, ...])[0]
        return augmented, new_sample_rate

class BackgroundNoise(Noise):
    def __init__(self, dir, uri, filename, local, filepath, db=20):
        super().__init__(dir, uri, filename, local, filepath)
        self.db = db
    
    def apply(self, waveform, sample_rate):
        noise, noise_sample_rate = self.get_sample()

        #print("sample rate: ", sample_rate, " noise sample rate: ", noise_sample_rate)
        new_sample_rate, waveform = self.downsample(waveform, sample_rate, noise_sample_rate)
        noise_len = noise.shape[-1]
        waveform_len = waveform.shape[-1]

        # make sure both noise and waveform has the same length
        if noise_len >= waveform_len: 
            noise = noise[:, :waveform.shape[-1]]
        else:
            diff = waveform_len - noise_len
            nb_of_diffs = diff // noise_len
            remainder = diff % noise_len

            noise_ = noise.clone()
            while nb_of_diffs > 0:
                nb_of_diffs -=1
                noise_ = torch.concat((noise_, noise), dim=-1)
            noise_ = torch.concat((noise_, noise[:,:remainder]), dim=-1)
            noise = noise_


        waveform_power = waveform.norm(p=2)
        noise_power = noise.norm(p=2)

        snr = math.exp(self.db / 10)
        scale = snr * noise_power / waveform_power

        result = (scale * waveform + noise)/2
        return result, new_sample_rate

# ================================== GENERAL ================================== #
def clip_audio(waveform, sample_rate:int, interval:int =3):
    '''
    Parameters
    ----------------------------------------------
    waveform: Tensor, the audio to be clipped
    sample_rate: int, the sample rate of this audio
    interval: int, how many seconds to clip from this audio

    Result
    ----------------------------------------------
    result: Tensor, the clipped audio
    '''
    assert interval >= 0, 'interval mustn\'t be negative'
    num_dims = len(list(waveform.shape))
    assert 2 >= num_dims > 0, 'waveform dimension should be greater than 0'
    if num_dims == 2:
        assert len(waveform[:]) <=2, 'number of channels can\'t be greater than 2'
    
    num_of_samples = None
    if num_dims == 2:
        num_of_samples = len(waveform[0, :])
    else:
        num_of_samples = len(waveform[:])

    total_number_of_seconds = num_of_samples / sample_rate
    #assert total_number_of_seconds >= interval, 'can\'t clip more than the audio length'  
    if total_number_of_seconds < interval:
        return waveform
    number_of_frames  = sample_rate * interval
    result = None
    if num_dims == 2:
        result = waveform[:,:number_of_frames]
    else:
        result = waveform[:number_of_frames]
    return result

def _list_all_filepaths(path):
    '''
    this function recursively returns paths in a 
    given a folder path

    Parameters
    -----------------------------------------------------
    path: str, the path to a folder

    Returns
    -----------------------------------------------------
    result: list, a list of all file paths inside 
            this folder
    '''

    # TODO: type checking
    assert type(path) is str, 'should be a string'
    # TODO: check the given path is a folder path
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith('.mp3') or file.endswith('.wav'):
                files.append(os.path.join(r, file))
    return files

# ================================== AUDIO IO OPERATIONS ================================== #
class AudioBackend:
    '''
    This module is used for saving and loading
    audio a given from filepath
    '''
    def save(self, filepath, audio, sampling_rate):
        # TODO: check type
        #torchaudio.backend.soundfile_backend.save(filepath, audio, sampling_rate)
        torchaudio.save(filepath, audio, sampling_rate)
    def load(self, filepath):
        # TODO: check type

        extension = filepath.split('/')[-1][-3:]
        if extension == 'mp3':
            return torchaudio.load(filepath)
        else:
            return torchaudio.backend.soundfile_backend.load(filepath)
            
# ================================== SHAZAM ================================== #
# TODO: file reader assume the files are in .mp3 format
def analyze_dir(path_to_dir):
    '''
    Parameters
    ------------------------------------------
    dir: str, the path to a directory with mp3 files
    '''
    djv.fingerprint_directory(path_to_dir, [".mp3"], 3)



def find_match(filepath:str):
  '''
  Parameters
  ------------------------------------
  filepath: str, the path to the audio
  '''
  song = djv.recognize(FileRecognizer, filepath)
  return song

def reset():
    djv.empty()
