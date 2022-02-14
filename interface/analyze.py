import os
from .utils import find_match, _list_all_filepaths
import wandb

TEST_DIR = '/Users/mohamedelfatih/Documents/baamtu/dejavu/augmented_audios'
noise_types = [
     'clean',
 'RVB2014_type1_noise_largeroom1_1',
'RVB2014_type2_noise_simroom1_1',
'RVB2014_type2_rir_simroom1_far_angla',
'RWCP_type1_rir_circle_ane_imp000',
'RWCP_type2_rir_cirline_ofc_imp130',
'RWCP_type4_rir_000l',
'RWCP_type4_rir_p50r',
'RWCP_type5_noise_cirline_ofc_ambient1'
    ]
if __name__ == '__main__':
    wandb.init(project="shazam")
    # TODO: check them
    # hyper-parameters
    hyper_parameters  = {
        'idx_freq_i':0,
        'idx_time_j':1,
        'default_fs':44100,
        'default_window_size':4096,
        'default_overlap_ratio':.5,
        'default_fan_value':15,
        'default_amp_min':20,
        'peak_neighborhood_size':20, 
        'min_hash_time_delta':0,
        'max_hash_time_delta':200,
        'peak_sort':True,
        'fingerprint_reduction':20
      }

    columns = ['noise type', 'average time to recognize', 'accuracy']
    table = wandb.Table(columns=columns)
    for noise_type in noise_types:
        dir_path = os.path.join(TEST_DIR, noise_type)
        filepaths = _list_all_filepaths(dir_path)
        counter = 0
        correct = 0
        mean_time= 0
        miss_table = wandb.Table(
        columns = ['filename', 'miss classified as']
      )
        for filepath in filepaths:
            counter +=1
            result = find_match(filepath)
            results = result['results']
            if len(results) == 0:
                print("match not found")
                miss_table.add_data(label, 'match not found')
                continue
            time = result['total_time']
            mean_time += time
            prediction = result['results'][0]['song_name']
            prediction = prediction.decode('utf-8')
            label = filepath.split('/')[-1][:-4]
            correct += int(label == prediction)
            if label != prediction:
                miss_table.add_data(label, prediction)
            print(f'current accuracy {correct / counter * 100 :.2f}')
        accuracy = correct / counter
        table.add_data(noise_type, mean_time / counter, accuracy)
        wandb.log({f"miss table for {noise_type}":miss_table}) 
    wandb.log({'table 1': table})    
