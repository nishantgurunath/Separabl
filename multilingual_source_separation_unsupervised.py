import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn




class multilingual_speech_model(nn.Module):


    def __init__(self):
        super(multilingual_speech_model, self).__init__()
        self.input_size = 321
        self.ehidden_size = 512
        self.dinput_size = 320 # 5x64
        self.dhidden_size = 512 
        self.mult_dim = 2 
        
        self.encoder = nn.LSTM(self.input_size,self.ehidden_size, num_layers = 2, bidirectional=True) # S x N x 512*2
        self.s_mu = nn.Linear(self.ehidden_size*2, int(self.dinput_size/5)) # S X N X 64
        self.s_sigma = nn.Linear(self.ehidden_size*2, int(self.dinput_size/5)) # S X N X 64
        self.r_mu = nn.Linear(self.ehidden_size*2, int(self.dinput_size/5)) # S X N X 64
        self.r_sigma = nn.Linear(self.ehidden_size*2, int(self.dinput_size/5)) # S X N X 64
        self.r1_mu = nn.Linear(self.ehidden_size*2, int(self.dinput_size/5)) # S X N X 64
        self.r1_sigma = nn.Linear(self.ehidden_size*2, int(self.dinput_size/5)) # S X N X 64
        self.r2_mu = nn.Linear(self.ehidden_size*2, int(self.dinput_size/5)) # S X N X 64
        self.r2_sigma = nn.Linear(self.ehidden_size*2, int(self.dinput_size/5)) # S X N X 64
    
        self.decoder = nn.LSTM(self.dinput_size, self.dhidden_size, bidirectional = True) # S x N x 512*2
        if(torch.cuda.is_available()):
            self.c = nn.Parameter(torch.zeros(2,self.dhidden_size).type(torch.cuda.FloatTensor))
            self.h = nn.Parameter(torch.zeros(2,self.dhidden_size).type(torch.cuda.FloatTensor))
        else:
            self.c = nn.Parameter(torch.zeros(2,self.dhidden_size))
            self.h = nn.Parameter(torch.zeros(2,self.dhidden_size))

        self.context = nn.Linear(self.dhidden_size*2, int(self.dinput_size/5)) # S x N x 64

        self.feature = nn.Linear(self.dhidden_size*2, self.dhidden_size*4) # S x N x 2048
        self.activation = nn.Tanh()
        self.scoring = nn.Linear(self.dhidden_size*4, self.input_size) # S x N x 321
   
    def encoder_block(self,inp,eps=0,eta=1,test=0): 
        
        x = rnn.pack_sequence(inp) 
        x,(h,c) = self.encoder(x)
        x,l = rnn.pad_packed_sequence(x) # S x N x 512*2
        S,N,E = x.shape  

        # Mask
        mask = torch.zeros((N,S)).type(torch.cuda.FloatTensor) # N x S
        for i in range(len(l)):
            mask[i,0:l[i]] = torch.ones(l[i]) 
        mask = torch.transpose(mask,0,1) # S x N

        ## z1 ##
        mu_s = self.s_mu(x) # S x N x 64
        log_var_s = self.s_sigma(x) # S x N x 64
        std_s = torch.exp(0.5*log_var_s)
        zs_embed = (mu_s+eps) + eta*std_s*(torch.randn(std_s.shape).type(torch.cuda.FloatTensor)) 
        z_s = torch.mean(zs_embed*mask.contiguous().view(S,N,1),dim=0)
 
        ## z2 ##
        mu_r = self.r_mu(x) # S x N x 64
        log_var_r = self.r_sigma(x) # S x N x 64
        std_r = torch.exp(0.5*log_var_r)
        z_r = (mu_r+eps) + eta*std_r*(torch.randn(std_r.shape).type(torch.cuda.FloatTensor)) 
        
        ## z3 ##
        mu_r1 = self.r1_mu(x) # S x N x 64
        log_var_r1 = self.r1_sigma(x) # S x N x 64
        std_r1 = torch.exp(0.5*log_var_r1)
        z_r1 = (mu_r1+eps) + eta*std_r1*(torch.randn(std_r1.shape).type(torch.cuda.FloatTensor)) 

        ## z3 ##
        mu_r2 = self.r2_mu(x) # S x N x 64
        log_var_r2 = self.r2_sigma(x) # S x N x 64
        std_r2 = torch.exp(0.5*log_var_r2)
        z_r2 = (mu_r2+eps) + eta*std_r2*(torch.randn(std_r2.shape).type(torch.cuda.FloatTensor)) 

        return zs_embed,z_s,mu_s,log_var_s,z_r,mu_r,log_var_r,z_r1,mu_r1,log_var_r1,z_r2,mu_r2,log_var_r2,mask


    def decoder_block(self,z_s,z_r,z_r1,z_r2):
        S,N,E = z_s.shape
        c = self.h.view(-1).unsqueeze(0) # 1 x 512*2
        c = c.expand(N,-1) # N x 512*2
        c = self.context(c) # N x 64
        inp = torch.cat((z_s[0],z_r[0],z_r1[0],z_r2[0],c), dim=1).unsqueeze(0) # 1 x N x 320
        h_0 = self.h.unsqueeze(1) # 2 x 1 x 512
        h_0 = h_0.expand(-1,N,-1).contiguous() # 2 x N x 512
        c_0 = self.c.unsqueeze(1) # 2 x 1 x 512
        c_0 = c_0.expand(-1,N,-1).contiguous() # 2 x N x 512
       
        x_n,(h_n,c_n) = self.decoder(inp,(h_0,c_0)) 

        out = x_n

        for i in range(1,len(z_s)): 
            c = x_n.view(N,-1) # N x 512*2
            c = self.context(c) # N x 64
            inp = torch.cat((z_s[i],z_r[i],z_r1[i],z_r2[i],c), dim=1).unsqueeze(0) # 1 x N x 320
            x_n,(h_n,c_n) = self.decoder(inp,(h_n,c_n))
            out = torch.cat((out,x_n), dim=0)


        # out S x N x 512*2
        return out



    def forward(self,inp,eps=0,eta=1,test=0):
        S = len(inp[0])
        N = len(inp)
        zs_embed,z_s,mu_s,log_var_s,z_r,mu_r,log_var_r,z_r1,mu_r1,log_var_r1,z_r2,mu_r2,log_var_r2,mask = self.encoder_block(inp,eps,eta,test)


        out = self.decoder_block(zs_embed,z_r,z_r1,z_r2)
        out = self.feature(out)
        out = self.activation(out)
        out = self.scoring(out)


        return out,mu_s,log_var_s,mu_r,log_var_r,mu_r1,log_var_r1,mu_r2,log_var_r2,mask
    

 
