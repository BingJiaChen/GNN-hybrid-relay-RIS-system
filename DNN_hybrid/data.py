from utils import *
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataLoader(Dataset):
    def __init__(self,M,N,L,batch_size):
        super(MyDataLoader,self).__init__()
        self.M = M
        self.N = N
        self.L = L
        self.batch_size = batch_size
        self.LOS_bs_ris = gen_LOS(self.N,self.M,10)
        self.LOS_bs_relay = gen_LOS(self.L,self.M,0)
        self.LOS_relay_ris = gen_LOS(self.N,self.L,10)

    def load_data(self,K,sigma=0):
        self.K = K
        if sigma == 0:
            H, C, channel_relay_user, channel_bs_user, relay_channel, relay_SINR_channel = generate_channel(self.M,self.N,self.L,self.K,self.batch_size,self.LOS_bs_ris,self.LOS_bs_relay,self.LOS_relay_ris,sigma)
        else:
            H, C, channel_relay_user, channel_bs_user, relay_channel, relay_SINR_channel ,perfect_channel_ris_user, perfect_channel_relay_user, perfect_H, perfect_C = generate_channel(self.M,self.N,self.L,self.K,self.batch_size,self.LOS_bs_ris,self.LOS_bs_relay,self.LOS_relay_ris,sigma)

        RIS_feature_1 = np.zeros((H.shape[0],self.M,self.N,2*self.K))
        RIS_feature_1[:,:,:,:self.K] = H.real
        RIS_feature_1[:,:,:,self.K:] = H.imag
        RIS_feature_1 = RIS_feature_1.transpose((0,3,1,2))

        user_feature_1 = np.zeros((H.shape[0],self.K,2,self.M,self.N))
        for k in range(self.K):
            temp = H[:,:,:,k]
            user_feature_1[:,k,0,:,:] = temp.real
            user_feature_1[:,k,1,:,:] = temp.imag

        RIS_feature_2 = np.zeros((C.shape[0],self.L,self.N,2*self.K))
        
        RIS_feature_2[:,:,:,:self.K] = C.real
        RIS_feature_2[:,:,:,self.K:] = C.imag
        RIS_feature_2 = RIS_feature_2.transpose((0,3,1,2))

        user_feature_2 = np.zeros((C.shape[0],self.K,2,self.L,self.N))
        for k in range(self.K):
            temp = C[:,:,:,k]
            user_feature_2[:,k,0,:,:] = temp.real
            user_feature_2[:,k,1,:,:] = temp.imag
        if sigma == 0:
            return torch.Tensor(RIS_feature_1), torch.Tensor(user_feature_1), torch.Tensor(RIS_feature_2), torch.Tensor(user_feature_2), H, C, channel_relay_user, channel_bs_user, relay_channel, relay_SINR_channel, channel_relay_user, H, C
        else:
            return torch.Tensor(RIS_feature_1), torch.Tensor(user_feature_1), torch.Tensor(RIS_feature_2), torch.Tensor(user_feature_2), H, C, channel_relay_user, channel_bs_user, relay_channel, relay_SINR_channel, perfect_channel_relay_user, perfect_H, perfect_C



if __name__ == '__main__':
    M = 8
    N = 50
    L = 4
    K = 4
    batch_size = 64
    num_sample = 64
    sigma_2 = 0.1
    dataloader = MyDataLoader(M, N, L, K, batch_size)
    RIS_node_1, user_node_1, RIS_node_2, user_node_2, H, C, channel_relay_user, channel_bs_user, relay_channel = dataloader.load_data()
    
    print(user_node_1.shape)
