import os
import random
from .utils import _list_all_filepaths

audio_dir = './downsampled_audios/'
if __name__ == '__main__':
    filepaths = _list_all_filepaths(audio_dir)
    print(len(filepaths))
    collection = {}
    counter = 20
    for filepath in filepaths:
        filename = filepath.split('/')[-1][:-4]
        index_to_split = -1
        index_counter = 0
        while filename[index_to_split] != '_' or index_counter == 0:
            if filename[index_to_split] == '_':
                index_counter +=1
            index_to_split -=1
        
        filename = filename[:index_to_split]
        if filename not in collection:
            collection[filename] = []
            collection[filename].append(filepath)
        else:
            collection[filename].append(filepath)
    nb_audios = 0
    for k, val in collection.items():
        nb_audios += len(val)
        print(f'Cheikh {k} has {len(val)}')
    print(f'number of audios: {nb_audios}')

    nb_audios_to_delete = 20
    for k, val in collection.items():
        nb_audios += len(val)
        # randomly choose {nb_audios_to_delete}
        to_delete = random.sample(val, nb_audios_to_delete)
        # loop through them and delete
        for filepath in to_delete:
            os.remove(filepath)
    
    print(f'Number of audios remaining {len(_list_all_filepaths(audio_dir))}')
    
        

