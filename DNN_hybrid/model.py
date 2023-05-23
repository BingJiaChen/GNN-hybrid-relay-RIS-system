import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class DNN(nn.Module):
    def __init__(self,M,N,K,Pt):
        super(DNN,self).__init__()
        self.M = M
        self.N = N
        self.K = K
        self.Pt = Pt

        self.PhaseNet = nn.Sequential(
            nn.Conv2d(2*self.K,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*self.M*(self.N+1),4*self.N),
            nn.BatchNorm1d(4*self.N),
            nn.ReLU(),
            nn.Linear(4*self.N,2*self.N)
        )

        self.BeamNet = nn.Sequential(
            nn.Linear(2*self.K*self.M,100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100,2*self.M*self.K)
        )

    def forward(self,x,H,d):
        batch_size = x.shape[0]
        d = torch.Tensor(np.concatenate((d.reshape((batch_size,1,self.K,-1)).real,d.reshape((batch_size,1,self.K,-1)).imag),axis=1)).to('cuda')
        d = d.reshape((batch_size,2*self.K,-1,1))
        x = torch.cat((x,d),dim=3)
        phase = self.PhaseNet(x)
        phase_re = phase[:,:self.N].unsqueeze(2)
        phase_im = phase[:,self.N:].unsqueeze(2)
        phase = torch.cat((phase_re,phase_im),dim=2)
        Theta = F.normalize(phase,dim=2)

        phase_re = Theta[:,:,0]
        phase_im = Theta[:,:,1]
        cas_channel = H
        num_sample = H.shape[0]
        rate = []
        phase_T = torch.cat((phase_re,phase_im),dim=1).unsqueeze(2)
        effective_channel = []

        for k1 in range(self.K):
            A_tmp_tran = np.transpose(cas_channel[:,:,:,k1],(0,2,1))
            A_tmp_real1 = np.concatenate([A_tmp_tran.real,A_tmp_tran.imag],axis=2)
            A_tmp_real2 = np.concatenate([-A_tmp_tran.imag,A_tmp_tran.real],axis=2)
            A_tmp_real = np.concatenate([A_tmp_real1,A_tmp_real2],axis=1)
            A_T_k = torch.Tensor(A_tmp_real).to('cuda')

            phase_A_T_k = torch.bmm(A_T_k.transpose(2,1),phase_T)

            if k1 == 0:
                effective_channel = phase_A_T_k.squeeze()
            else:
                effective_channel = torch.cat((effective_channel,phase_A_T_k.squeeze()),dim=1)

        W = self.BeamNet(effective_channel)
        W = W.reshape((num_sample,2*self.M,self.K))
        # W = torch.transpose(W,2,1)
        W = F.normalize(W,dim=2)*np.sqrt(self.Pt)/self.K

        return Theta, W

class DNN_module(nn.Module):
    def __init__(self,M,N,L,K,Pt):
        super(DNN_module,self).__init__()
        self.M = M
        self.N = N
        self.K = K
        self.L = L
        self.Pt = Pt

        self.phase_I = DNN(M,N,K,Pt)
        self.phase_II = DNN(L,N,K,Pt)

    def forward(self, x1, H , d1 , x2, C , d2):
        Theta1, W = self.phase_I(x1,H,d1)
        Theta2, F = self.phase_II(x2,C,d2)

        return Theta1, W, Theta2, F


