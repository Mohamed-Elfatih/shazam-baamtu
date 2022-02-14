# import os
# import uuid
# from flask import Flask, flash, request, redirect
from interface.utils import find_match, AudioBackend
import torch
import librosa
import numpy as np
# UPLOAD_FOLDER = './source/files'

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# @app.route('/')
# def root():
#     return app.send_static_file('index.html')


# @app.route('/save-record', methods=['POST'])
# def save_record():
#     # check if the post request has the file part
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
#     file = request.files['file']
#     # if user does not select file, browser also
#     # submit an empty part without filename
#     if file.filename == '':
#         flash('No selected file')
#         return redirect(request.url)
#     file_name = str(uuid.uuid4()) + ".wav"
#     full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
#     file.save(full_file_name)
#     result = find_match(full_file_name)
#     print("result >> ", result)
#     return '<h1>Success</h1>'


# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=43800, debug=True)
import gradio as gr
import torch
import zipfile
#from pyctcdecode import build_ctcdecoder
#from speechbrain.pretrained import EncoderASR
#from transformers.file_utils import cached_path, hf_bucket_url

# cache_dir = './cache/'
# lm_file = hf_bucket_url(
#     "dragonSwing/wav2vec2-base-vn-270h", filename='4gram.zip')
# lm_file = cached_path(lm_file, cache_dir=cache_dir)
# with zipfile.ZipFile(lm_file, 'r') as zip_ref:
#     zip_ref.extractall(cache_dir)
# lm_file = cache_dir + 'lm.binary'
# vocab_file = cache_dir + 'vocab-260000.txt'
# model = EncoderASR.from_hparams(source="dragonSwing/wav2vec2-base-vn-270h",
#                                 savedir="./pretrained/wav2vec-vi-asr"
#                                 )


# def get_decoder_ngram_model(tokenizer, ngram_lm_path, vocab_path=None):
#     unigrams = None
#     if vocab_path is not None:
#         unigrams = []
#         with open(vocab_path, encoding='utf-8') as f:
#             for line in f:
#                 unigrams.append(line.strip())

#     vocab_dict = tokenizer.get_vocab()
#     sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
#     vocab = [x[1] for x in sort_vocab]
#     vocab_list = vocab

    # convert ctc blank character representation
    # vocab_list[tokenizer.pad_token_id] = ""
    # # replace special characters
    # vocab_list[tokenizer.word_delimiter_token_id] = " "
    # # specify ctc blank char index, since conventially it is the last entry of the logit matrix
    # decoder = build_ctcdecoder(vocab_list, ngram_lm_path, unigrams=unigrams)
    # return decoder


# ngram_lm_model = get_decoder_ngram_model(model.tokenizer, lm_file, vocab_file)


# def transcribe_file(path, max_seconds=20):
#     waveform = model.load_audio(path)
#     if max_seconds > 0:
#         waveform = waveform[:max_seconds*16000]
#     batch = waveform.unsqueeze(0)
#     rel_length = torch.tensor([1.0])
#     with torch.no_grad():
#         logits = model(batch, rel_length)
#     text_batch = [ngram_lm_model.decode(
#         logit.detach().cpu().numpy(), beam_width=500) for logit in logits]
#     return text_batch[0]


def speech_recognize(file_mic):
    print("am i called")
    audio_backend = AudioBackend()
    if file_mic is not None:
        filepath = file_mic
    else:
        return ""
    # text = model.transcribe_file(file)
    # text = transcribe_file(file)
    waveform, fs = audio_backend.load(filepath)
    print(filepath)
    # downsample
    frequency=12000
    waveform = librosa.resample(np.array(waveform, dtype=np.float32), fs, frequency, res_type='kaiser_best')
    print(find_match(filepath))
    fs = frequency
    waveform = torch.tensor(waveform).float()
    # save
    audio_backend.save('test.mp3', waveform, fs)
    return find_match('test.mp3')

def dummy():
    pass
inputs = gr.inputs.Audio(
    source="microphone", type='filepath', optional=True)
outputs = gr.outputs.Textbox(label="Output Text")
title = "Cheikh Detection"
description = "detect the reciting chiekh"
article = "<p style='text-align: center'><a href='https://huggingface.co/dragonSwing/wav2vec2-base-vn-270h' target='_blank'>Pretrained model</a></p>"
# examples = [
#     ['example1.wav', 'example1.wav'],
#     ['example2.mp3', 'example2.mp3'],
#     ['example3.mp3', 'example3.mp3'],
#     ['example4.wav', 'example4.wav'],
# ]
gr.Interface(speech_recognize, inputs=inputs, outputs=outputs, title=title,
             description=description, article=article,).launch()
