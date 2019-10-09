import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn




class multilingual_speech_model(nn.Module):


    def __init__(self,num_latent):
        super(multilingual_speech_model, self).__init__()
        self.num_latent = num_latent 
        self.input_size = 321
        self.ehidden_size = 512
        self.dinput_size = (num_latent+1)*64
        self.dhidden_size = 512 
        self.mult_dim = 2
        
        
        self.encoder = nn.LSTM(self.input_size,self.ehidden_size, num_layers = 2, bidirectional=True) # S x N x 512*2
        if(torch.cuda.is_available()):
            self.mu = nn.Module()
            self.sigma = nn.Module()
            for i in range(num_latent):
                self.mu.add_module(str(i),nn.Linear(self.ehidden_size*2, int(self.dinput_size/(num_latent+1))).cuda())
            for i in range(num_latent):
                self.sigma.add_module(str(i),nn.Linear(self.ehidden_size*2, int(self.dinput_size/(num_latent+1))).cuda())
        else:
            self.fc_gaussian = [(nn.Linear(self.ehidden_size*2, int(self.dinput_size/(num_latent+1))), \
                                 nn.Linear(self.ehidden_size*2, int(self.dinput_size/(num_latent+1))))]*num_latent

        self.decoder = nn.LSTM(self.dinput_size, self.dhidden_size, bidirectional = True) # S x N x 512*2
        if(torch.cuda.is_available()):
            self.c = nn.Parameter(torch.zeros(2,self.dhidden_size).type(torch.cuda.FloatTensor))
            self.h = nn.Parameter(torch.zeros(2,self.dhidden_size).type(torch.cuda.FloatTensor))
        else:
            self.c = nn.Parameter(torch.zeros(2,self.dhidden_size))
            self.h = nn.Parameter(torch.zeros(2,self.dhidden_size))

        self.context = nn.Linear(self.dhidden_size*2, int(self.dinput_size/(num_latent+1))) # S x N x 64

        self.feature = nn.Linear(self.dhidden_size*2, self.dhidden_size*4) # S x N x 2048
        self.activation = nn.Tanh()
        self.scoring = nn.Linear(self.dhidden_size*4, self.input_size) # S x N x 321
   
    def encoder_block(self,inp,eps=0,eta=1,test=0): 
        
        x = rnn.pack_sequence(inp) 
        x,(h,c) = self.encoder(x)
        x,l = rnn.pad_packed_sequence(x) # S x N x 512*2
        S,N,E = x.shape  
        # Mask
        if torch.cuda.is_available():
            mask = torch.zeros((N,S)).type(torch.cuda.FloatTensor) # N x S
        else:
            mask = torch.zeros((N,S)).type(torch.FloatTensor) # N x S

        for i in range(len(l)):
            mask[i,0:l[i]] = torch.ones(l[i]) 
        mask = torch.transpose(mask,0,1) # S x N


        if torch.cuda.is_available():
            z = torch.zeros((self.num_latent,S,N,self.dinput_size//(self.num_latent+1))).type(torch.cuda.FloatTensor)
            mu = torch.zeros((self.num_latent,S,N,self.dinput_size//(self.num_latent+1))).type(torch.cuda.FloatTensor)
            log_var = torch.zeros((self.num_latent,S,N,self.dinput_size//(self.num_latent+1))).type(torch.cuda.FloatTensor)
        else:
            z = torch.zeros((self.num_latent,S,N,self.dinput_size//(self.num_latent+1)))
            mu = torch.zeros((self.num_latent,S,N,self.dinput_size//(self.num_latent+1)))
            log_var = torch.zeros((self.num_latent,S,N,self.dinput_size//(self.num_latent+1)))

        for i,fc in enumerate(self.mu._modules.values()):
            mu[i] = fc(x)
            
        for i,fc in enumerate(self.sigma._modules.values()):
            log_var[i] = fc(x)

        for i in range(self.num_latent):
            mu_temp = mu[i]           
            log_var_temp = log_var[i]
            std_temp = torch.exp(0.5*log_var_temp)
            if torch.cuda.is_available():
                z_temp = (mu_temp+eps) + eta*std_temp*(torch.randn(std_temp.shape).type(torch.cuda.FloatTensor))
            else:
                z_temp = (mu_temp+eps) + eta*std_temp*(torch.randn(std_temp.shape).type(torch.FloatTensor))

            z[i] = z_temp 
        #print (self.mu._modules)
        #print (self.mu._modules['0'])
        #mu = self.mu._modules['0'](x) # S x N x 64
        #log_var = self.sigma._modules['0'](x) # S x N x 64
        #std = torch.exp(0.5*log_var)
        #z = (mu+eps) + eta*std*(torch.randn(std.shape).type(torch.cuda.FloatTensor)) 


        return z,mu,log_var,mask


    def decoder_block(self,z):
        K,S,N,E = z.shape
        #S,N,E = z.shape
        c = self.h.view(-1).unsqueeze(0) # 1 x 512*2
        c = c.expand(N,-1) # N x 512*2
        c = self.context(c) # N x 64
        inp = z.permute(1,2,0,3)[0].contiguous().view(N,K*E)
        inp = torch.cat((inp,c),dim=1).unsqueeze(0)
        #inp = torch.cat((z[0],c),dim=1).unsqueeze(0)
        h_0 = self.h.unsqueeze(1) # 2 x 1 x 512
        h_0 = h_0.expand(-1,N,-1).contiguous() # 2 x N x 512
        c_0 = self.c.unsqueeze(1) # 2 x 1 x 512
        c_0 = c_0.expand(-1,N,-1).contiguous() # 2 x N x 512
       
        x_n,(h_n,c_n) = self.decoder(inp,(h_0,c_0)) 

        out = x_n

        for i in range(1,S): 
            c = x_n.view(N,-1) # N x 512*2
            c = self.context(c) # N x 64
            inp = z.permute(1,2,0,3)[i].contiguous().view(N,K*E)
            inp = torch.cat((inp,c),dim=1).unsqueeze(0)
            #inp = torch.cat((z[i],c),dim=1).unsqueeze(0)
            x_n,(h_n,c_n) = self.decoder(inp,(h_n,c_n))
            out = torch.cat((out,x_n), dim=0)


        return out



    def forward(self,inp,eps=0,eta=1,test=0):
        S = len(inp[0])
        N = len(inp)
        z,mu,log_var,mask = self.encoder_block(inp,eps,eta,test)

        out = self.decoder_block(z)
        out = self.feature(out)
        out = self.activation(out)
        out = self.scoring(out)

        return out,mu,log_var,mask
    

if __name__ == "__main__":
    x = torch.zeros((300,321))
    y = torch.ones((300,321))
    z = torch.ones((300,321))*2
    x = [x,y,z]
    model = multilingual_speech_model(5)
    print (torch.cuda.is_available())
    print (model(x)[0].shape) 
