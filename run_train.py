import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from sklearn.mixture import GaussianMixture
import argparse
from model import multilingual_speech_model
import os


## Parameters
# Data - 3 sec, 321x300, sr=16000, 100 frames/sec 


def kl_anneal_function(step, k=0.0050, x0=4000, anneal_function='logistic'):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)

## Dataloader Class
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
        data = self.data
        index = np.arange(len(data))
        if(self.shuffle==True):
            np.random.shuffle(index)
        data = data[index]
        num_batches = len(data)//self.batch_size
        print (num_batches)
        num_res = len(data)%self.batch_size
        num_utt = num_batches*self.batch_size

        i = 0
        while(i<num_batches):
            out_data = []
            s = i*self.batch_size
            e = (i+1)*self.batch_size
            index1 = np.arange(s,e)
            lengths = [len(data[j]) for j in range(s,e)] 
            min_len = min(min(lengths)-1,300)

            for j in range(s,e):
                k = np.random.randint(0,len(data[j])-min_len)
                if torch.cuda.is_available():
                    out_data.append(torch.cuda.FloatTensor(data[j][k:k+min_len]))
                else:
                    out_data.append(torch.FloatTensor(data[j][k:k+min_len]))
            yield out_data
            i = i + 1
            

        ## Residual Data
        if(num_res > 0):
            out_data = []
            s = num_utt
            e = len(data)
            index1 = np.arange(s,e)
            lengths = [len(data[j]) for j in range(s,e)] 
            min_len = min(min(lengths)-1,300)

            for j in range(s,e):
                k = np.random.randint(0,len(data[j])-min_len)
                if torch.cuda.is_available():
                    out_data.append(torch.cuda.FloatTensor(data[j][k:k+min_len]))
                else:
                    out_data.append(torch.FloatTensor(data[j][k:k+min_len]))
            yield out_data




class model_run:

    def train(self, num_epochs, num_latent, data):

        ## Parameters ##
        batch_size = 16
        lr = 0.001
        print (len(data))

        ## Model ##
        model = multilingual_speech_model(num_latent)
        model = model.cuda() if torch.cuda.is_available() else model
        #model_state = torch.load('models_dc/model49.pt')
        #model.load_state_dict(model_state)    


        ##Data Loader   ## 
        loader = SpeechModelDataLoader(abs(data), shuffle=True, batch_size=batch_size)


        # Criterion - Negative Log Likelihood + KL Divergence Z
        criterion_mse = nn.MSELoss(reduction = 'none')
               
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
        updates = 0
        for e in range(num_epochs):
            epoch_loss = 0
            epoch_KL = 0
            epoch_MSE = 0
            model.train()
            i = 0
            samples = 0
            for data in loader:
                optimizer.zero_grad()
                padded_data = rnn.pad_sequence(data) 
                S,N,E = padded_data.shape
                
                out,mu,log_var,mask = model(data)
                
                samples += N
                updates += 1

                out = out.view(-1,E)*mask.contiguous().view(-1,1)

                K,S,N,E1 = mu.shape
                mu = mu.permute(1,2,0,3).contiguous().view(-1,K,E1)*mask.contiguous().view(-1,1,1)
                mu = mu.permute(1,0,2)
                log_var = log_var.permute(1,2,0,3).contiguous().view(-1,K,E1)*mask.contiguous().view(-1,1,1)
                log_var = log_var.permute(1,0,2)

                KL,MSE = self.loss_fn(out,padded_data.view(-1,E),mu,log_var,criterion_mse)
                MSE = MSE/(S) 
                weight =  kl_anneal_function(updates,k=40*batch_size/(len(data)*num_epochs),x0=len(data)*num_epochs//(batch_size*2))
                NLL =  MSE + (torch.sum(KL))*weight
                NLL.backward()
                epoch_loss += NLL.item()
                epoch_KL += torch.sum(KL).item()/K
                epoch_MSE += MSE.item()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                i = i + 1
                if(i%10==0):
                    print("Epoch: ", e, "Iter: ", i, "Loss: ", (epoch_loss/(samples)), "KL", epoch_KL/samples,  "MSE ", epoch_MSE/samples)

           
            if epoch_KL/samples < 60 and epoch_MSE/samples < 250:
                torch.save(model.state_dict(), "models/model" + str(e) + ".pt")
                print("Epoch: ", e,  "Loss: ", (epoch_loss/(samples)), "KL", epoch_KL/samples,  "MSE ", epoch_MSE/samples)
                #break 


    def loss_fn(self,recon_x, x, mu, log_var, criterion_mse):
        MSE = criterion_mse(recon_x, x)
        MSE = torch.sum(MSE)

        K,N,E = mu.shape
        mu = mu.contiguous().view(K,-1)
        log_var = log_var.contiguous().view(K,-1)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1)
        
        return KLD, MSE



if __name__ == "__main__":

    ## Parse Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Path to Train Data")
    args = vars(ap.parse_args())

    ## Load Data
    train_data = np.load(args['path'])
    #train_data = np.array([np.zeros((300,321))]*32)
    num_epochs = 50

    ## Find No. of latent nodes
    K = 2
    while(1):
        gmm = GaussianMixture(n_components=K)
        gmm.fit(train_data)
        likelihood = gmm.lower_bound_
        if(likelihood > 1070):
            break
        K += 1

    directory = "./models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    ## Train
    run = model_run()
    run.train(num_epochs, K-1, train_data)

