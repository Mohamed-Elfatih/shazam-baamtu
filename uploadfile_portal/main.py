from fastapi import FastAPI, UploadFile, File
from interface.utils import find_match, AudioBackend
import torch
import librosa
import numpy as np
import shutil




app = FastAPI()
@app.post("/")
async def root(file: UploadFile = File(...)):
    with open('temp.mp3', 'wb') as buffer:
        shuilt.copyfileobj(file.file, buffer)
    print(find_match('temp.mp3'))
    return {"file_name": file.path, "result":str(find_match('temp.mp3'))}
