from fastapi import FastAPI, UploadFile, File
from interface.utils import find_match, AudioBackend
import torch
import librosa
import numpy as np
import shutil




app = FastAPI()
@app.post("/")
async def root(file: UploadFile = File(...)):
    # saving the file temporary in the system
    with open('temp.mp3', 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = find_match('temp.mp3')['results']
    if len(results) == 0:
        return {'result': 'no match found'}
    
    best_result = results[0]
    result_name = best_result['song_name'] 
    
    def parse_surat(filename):
        # TODO: the type of filename must be str
        filename = str(filename)
        splitted = filename.split('_')
        surat_no = int(splitted[-2])
        chiekh  = '_'.join(splitted[:-2])

        #chiekh  = " ".join(chiekh.split(' ')[:-1]) 
        return chiekh, surat_no
    
    chiekh, surat_no = parse_surat(result_name)

    # TODO: make sure the sampling frequency is 11500Hz

    # TODO: should delete the temp file

    # TODO: parse the result

    # TODO: create a function to get surate name

    return {"Chiekh":chiekh, 'Surat No':surat_no}
