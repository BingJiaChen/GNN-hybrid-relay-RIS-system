import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from utils import *
from model import *
import matplotlib.pyplot as plt
import time

class Trainer():
    def __init__(self,M,N,L,training_K,testing_K,batch_size,sigma):
        self.M = M
        self.N = N
        self.training_K = training_K
        self.testing_K = testing_K
        self.K = training_K
        self.L = L
        self.b = 2
        self.batch_size = batch_size
        self.dataloader = MyDataLoader(self.M,self.N,self.L,self.batch_size)
        self.device = 'cuda'
        self.model = DNN_module(self.M,self.N,self.L,self.K,0.01).to(self.device)
        self.n_iter = 2000
        self.log_interval = 10
        self.log_eval_interval = 100
        self.threshold = 0.01
        self.sigma = sigma


    def train_batch(self):
        self.model.train()
        RIS_node_1, user_node_1, RIS_node_2, user_node_2, H, C, channel_relay_user, channel_bs_user, relay_channel, relay_SINR_channel, perfect_channel_relay_user, perfect_H, perfect_C = self.dataloader.load_data(self.training_K)
        RIS_node_1 = RIS_node_1.to(self.device)
        user_node_1 = user_node_1.to(self.device)
        RIS_node_2 = RIS_node_2.to(self.device)
        user_node_2 = user_node_2.to(self.device)
        
        self.opt.zero_grad()
        Theta1, W, Theta2, Relay = self.model(RIS_node_1,H,channel_bs_user,RIS_node_2,C,channel_relay_user)
        
        Bits = 2**self.b

        loss, sum_rate, con = cal_loss(W, Relay, Theta1, Theta2, H, C, channel_relay_user, channel_bs_user, relay_channel,self.threshold, relay_SINR_channel)

        loss.backward()
        self.opt.step()

        return loss.item(), sum_rate.item(), con

    def train(self):
        self.opt = torch.optim.Adam(self.model.parameters(),lr=0.001,weight_decay=1e-6)
        best_loss = 0
        train_loss = []
        train_total_loss = []
        train_con = []
        train_total_sumrate = []
        train_sum_rate = []
        test_sample = 5120
        val_total = []
        val_total_con = []
        for i in range(self.n_iter):
            loss, sum_rate, con = self.train_batch()
            train_loss.append(loss)
            train_sum_rate.append(sum_rate)
            train_con.append(con)
            if i%self.log_interval==0:
                print(f"[Train | {i}/{self.n_iter} ] loss = {np.mean(train_loss):.5f}, sum rate = {np.mean(train_sum_rate):.5f}, over threshold = {np.mean(train_con):.3f}")
                train_total_sumrate.append(np.mean(train_sum_rate))
                train_total_loss.append(-np.mean(train_loss))
                val_total_con.append(np.mean(train_con))
                train_loss = []
                train_sum_rate = []
                train_con = []

            if (i+1)%self.log_eval_interval==0:
                sum_rate, total_con = self.eval(self.dataloader,test_sample)
                print(f"[Val | {i+1}/{self.n_iter} ] sum rate = {(sum_rate):.5f}, over threshold = {total_con:.3f}%")
                
                val_total.append(sum_rate)
                val_total_con.append(total_con)

        plt.show()
    def eval(self,dataloader,test_sample):
        self.model.eval()
        iteration = int(test_sample/self.batch_size)
        val_loss = 0
        total_sample = 0
        total_con = 0
        sum_rate = 0
        num_bits = 2
        total_time = 0
        with torch.no_grad():
            for i in range(iteration):
                start_time = time.time()
                RIS_node_1, user_node_1, RIS_node_2, user_node_2, H, C, channel_relay_user, channel_bs_user, relay_channel, relay_SINR_channel, perfect_channel_relay_user, perfect_H, perfect_C = self.dataloader.load_data(self.testing_K,self.sigma)
                RIS_node_1 = RIS_node_1.to(self.device)
                user_node_1 = user_node_1.to(self.device)
                RIS_node_2 = RIS_node_2.to(self.device)
                user_node_2 = user_node_2.to(self.device)
                
                Theta1, W, Theta2, Relay = self.model(RIS_node_1,perfect_H,channel_bs_user,RIS_node_2,perfect_C,perfect_channel_relay_user)
                
                Theta1 = discrete_mapping(Theta1,num_bits)
                Theta2 = discrete_mapping(Theta2,num_bits)
                end_time = time.time()
                
                total_time += (end_time-start_time)
                


                loss, s, con = cal_loss(W, Relay, Theta1, Theta2, H, C, channel_relay_user, channel_bs_user, relay_channel, self.threshold, relay_SINR_channel)
                sum_rate += s.item()
                total_con += con
            
        return sum_rate/iteration/2, total_con/iteration

if __name__ == '__main__':
    batch_size = 512
    sigma = 0
    training_K = 4
    testing_K = 4
    M = 8
    N = 50
    L = 4
    
    trainer = Trainer(M,N,L,training_K,testing_K,batch_size,sigma)
    trainer.train()

    
    
    


