import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.utils.rnn as rnn
import argparse
from model import multilingual_speech_model
import os
import librosa
from rpca import *

def model_eval(audio_file, model):
    x,fs = librosa.load(audio_file,sr=16000)
    x = librosa.stft(x,n_fft=640)
    magx,P = librosa.magphase(x)

    out,mu,log_var,mask = model([torch.cuda.FloatTensor(magx.T)],eps=0,eta=0,test=0)
    S,N,E = out.shape
    out = (out.view(-1,E)*mask.contiguous().view(-1,1))
    out = out.double().cpu().detach().numpy()
    out = out.T
    x = out*P
    x,M = speech(x,0.3,0.5)
    y = librosa.istft(x)

    output_file = audio_file.split('/')[-1]
    output_file = output_file.split('.')[0]
    output_file = './output/' + output_file + '_cleaned.wav'
    librosa.output.write_wav(output_file,y,fs)


if __name__ == "__main__":

    ## Parse Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Path to Wav File")
    ap.add_argument("-m", "--model", required=True, help="Path to Trained Model")
    args = vars(ap.parse_args())

    ## Model
    model_state = torch.load(args['model'])
    num_latent = model_state['decoder.weight_ih_l0'].shape[1]//64 - 1
    print (num_latent)
    model = multilingual_speech_model(num_latent)
    model = model.cuda() if torch.cuda.is_available() else model
    model.load_state_dict(model_state)    


    ## Eval
    directory = "./output"
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_eval(args['path'],model) 
