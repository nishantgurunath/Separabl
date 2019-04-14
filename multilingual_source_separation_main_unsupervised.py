import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from multilingual_source_separation_unsupervised import *
from operator import itemgetter
import torch.nn.utils.rnn as rnn
import librosa
import os
from rpca import *

def loss_fn(recon_x, x, mu_s, logvar_s, mu_r, logvar_r, mu_r1, logvar_r1, mu_r2, logvar_r2, criterion_mse):
    MSE = criterion_mse(recon_x, x)
    MSE = torch.sum(MSE)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_s = -0.5 * torch.sum(1 + logvar_s - mu_s.pow(2) - logvar_s.exp())
    KLD_r = -0.5 * torch.sum(1 + logvar_r - mu_r.pow(2) - logvar_r.exp())
    KLD_r1 = -0.5 * torch.sum(1 + logvar_r1 - mu_r1.pow(2) - logvar_r1.exp())
    KLD_r2 = -0.5 * torch.sum(1 + logvar_r2 - mu_r2.pow(2) - logvar_r2.exp())

    ## CE
    return KLD_s, KLD_r, KLD_r1, KLD_r2, MSE

def kl_anneal_function(step, k=0.0050, x0=4000, anneal_function='logistic'):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)



class SpeechModelDataLoader(DataLoader):
    """
        TODO: Define data loader logic here
    """
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # concatenate your articles and build into batches
        data = self.data[0]
        labels = self.data[1]
        index = np.arange(len(data))
        if(self.shuffle==True):
            np.random.shuffle(index)
        data = data[index]
        labels = labels[index]
        num_batches = len(data)//self.batch_size
        print (num_batches)
        num_res = len(data)%self.batch_size
        num_utt = num_batches*self.batch_size

        i = 0
        while(i<num_batches):
            out_data = []
            out_labels = []
            s = i*self.batch_size
            e = (i+1)*self.batch_size
            index1 = np.arange(s,e)
            lengths = [len(data[j]) for j in range(s,e)] 
            min_len = min(min(lengths)-1,300)

            for j in range(s,e):
                k = np.random.randint(0,len(data[j])-min_len)
                out_data.append(torch.cuda.FloatTensor(data[j][k:k+min_len]))
                out_labels.append(labels[j])
            out_labels = torch.cuda.LongTensor(out_labels)
            yield out_data, out_labels
            i = i + 1
            

        ## Residual Data
        if(num_res > 0):
            out_data = []
            out_labels = []
            s = num_utt
            e = len(data)
            index1 = np.arange(s,e)
            lengths = [len(data[j]) for j in range(s,e)] 
            min_len = min(min(lengths)-1,300)

            for j in range(s,e):
                k = np.random.randint(0,len(data[j])-min_len)
                out_data.append(torch.cuda.FloatTensor(data[j][k:k+min_len]))
                out_labels.append(labels[j])
            out_labels = torch.cuda.LongTensor(out_labels)
            
            yield out_data, out_labels

def librosa_eval(file_name,model):
        x,fs = librosa.load(file_name,sr=16000)
        x = librosa.stft(x,n_fft=640)
        magx,P = librosa.magphase(x)
        y = librosa.istft(x)
        librosa.output.write_wav('soundo.wav',y,fs)

        out,mu_s,log_var_s,mu_r,log_var_r,mu_r1,log_var_r1,mu_r2,log_var_r2,mask = model([torch.cuda.FloatTensor(magx.T)],eps=0,eta=0,test=0)
        S,N,E = out.shape
        out = (out.view(-1,E)*mask.contiguous().view(-1,1))
        out = out.double().cpu().detach().numpy()
        out = out.T
        x = out*P
        x,M = speech(x,0.3,0.5)
        y = librosa.istft(x)

        librosa.output.write_wav('soundg.wav',y,fs)




class model_run:

    def train(self,num_epochs):

        ## Parameters ##
        batch_size = 16
        lr = 0.001

        ## Load Data##
        # Wilderness
        data = np.load('../data/train_data.npy')
        labels = np.load('../data/train_labels.npy')

        # Hub4
        #data = np.load('../Data/train_data_hub4.npy')
        #labels = np.load('../Data/train_labels_hub4.npy')
        #print (len(data), len(labels))
        ## Model ##
        model = multilingual_speech_model()
        model = model.cuda() if torch.cuda.is_available() else model
        #model_state = torch.load('models_dc/model49.pt')
        #model.load_state_dict(model_state)    


        ##Data Loader   ## 
        loader = SpeechModelDataLoader((abs(data),labels), shuffle=True, batch_size=batch_size)


        # Criterion - Negative Log Likelihood + KL Divergence Z
        criterion_mse = nn.MSELoss(reduction = 'none')
        criterion_ce = nn.CrossEntropyLoss(reduction = 'mean')
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
        updates = 0
        for e in range(num_epochs):
            epoch_loss = 0
            epoch_KL_s = 0
            epoch_KL_r = 0
            epoch_KL_r1 = 0
            epoch_KL_r2 = 0
            #epoch_CE = 0
            epoch_MSE = 0
            model.train()
            i = 0
            samples = 0
            for data, labels in loader:
                optimizer.zero_grad()
                padded_data = rnn.pad_sequence(data) 
                S,N,E = padded_data.shape
                
                out,mu_s,log_var_s,mu_r,log_var_r,mu_r1,log_var_r1,mu_r2,log_var_r2,mask = model(data)
                
                samples += N
                updates += 1

                out = out.view(-1,E)*mask.contiguous().view(-1,1)
                mu_s = mu_s.view(-1,mu_s.shape[2])*mask.contiguous().view(-1,1)
                log_var_s = log_var_s.view(-1,log_var_s.shape[2])*mask.contiguous().view(-1,1)
                mu_r = mu_r.view(-1,mu_r.shape[2])*mask.contiguous().view(-1,1)
                log_var_r = log_var_r.view(-1,log_var_r.shape[2])*mask.contiguous().view(-1,1)
                mu_r1 = mu_r1.view(-1,mu_r1.shape[2])*mask.contiguous().view(-1,1)
                log_var_r1 = log_var_r1.view(-1,log_var_r1.shape[2])*mask.contiguous().view(-1,1)
                mu_r2 = mu_r2.view(-1,mu_r2.shape[2])*mask.contiguous().view(-1,1)
                log_var_r2 = log_var_r2.view(-1,log_var_r2.shape[2])*mask.contiguous().view(-1,1)

                KL_s,KL_r,KL_r1,KL_r2,MSE = loss_fn(out,padded_data.view(-1,E),mu_s,log_var_s,mu_r,log_var_r,mu_r1,log_var_r1,mu_r2,log_var_r2,criterion_mse)
                MSE = MSE/(S) 
                weight =  kl_anneal_function(updates)
                NLL =  MSE + (KL_s + KL_r + KL_r1 + KL_r2)*weight
                NLL.backward()
                epoch_loss += NLL.item()
                epoch_KL_s += KL_s.item()
                epoch_KL_r += KL_r.item()
                epoch_KL_r1 += KL_r1.item()
                epoch_KL_r2 += KL_r2.item()
                epoch_MSE += MSE.item()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                i = i + 1
                if(i%10==0):
                    print("Epoch: ", e, "Iter: ", i, "Loss: ", (epoch_loss/(samples)), "KL_s", epoch_KL_s/samples, "KL_r", epoch_KL_r/samples,\
                          "KL_r1", epoch_KL_r1/samples, "KL_r2", epoch_KL_r2/samples, "MSE ", epoch_MSE/samples)

           
            if (e+1) % 1 == 0:
                torch.save(model.state_dict(), "models_dc/model" + str(e) + ".pt")
                print("Epoch: ", e, "Iter: ", i, "Loss: ", (epoch_loss/(samples)))

    def eval(self):

        # Load Model #
        model = multilingual_speech_model()
        model = model.cuda() if torch.cuda.is_available() else model
        model_state = torch.load('models_dc/hmodel4.pt')
        model.load_state_dict(model_state)    
        model.eval()
        model.cuda()
        # Load Data #
        #file_name = '../Data/B01___01_San_Mateo___ACCIBSN1DA_00036.wav'
        #file_name = '../data/B01___01_Matthew_____ADHBSUN2DA_00005.wav'
        #file_name = '../Data/i960711p_0_10.wav'
        file_name = '../../Hub4/l960710_87_97.wav'
        #file_name = '../data/ADHBSU/wav/B01___01_Matthew_____ADHBSUN2DA_00006.wav'
        #file_name = '../data/lazarus.mp3'
        #world(file_name,model)
        librosa_eval(file_name,model)



M = model_run
M.train(M,50)
#M.eval(M)
