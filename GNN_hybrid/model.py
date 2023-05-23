import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *



class node_update_layer(nn.Module):
    def __init__(self,in_dim,M,N,ch):
        super(node_update_layer,self).__init__()
        self.in_dim_RIS = in_dim*2
        self.in_dim_user = in_dim*3
        self.in_dim = in_dim
        self.ch = ch
        self.M = M
        self.N = N
        self.l1 = ch
        self.l2 = ch
        self.f3_RIS = nn.Sequential(
            nn.Linear(self.in_dim_RIS,self.l1),
            nn.ReLU(),
            nn.Linear(self.l1,self.l2)
        )

        self.f2_user = nn.Sequential(
            nn.Linear(self.in_dim_user,self.l1),
            nn.ReLU(),
            nn.Linear(self.l1,self.l2)
        )

        self.f0_user = nn.Linear(in_dim,2*M)
        self.f1_RIS = nn.Linear(in_dim,2*N)

    def forward(self,user_node,RIS_node,cas_channel,d_link):
        batch = user_node.shape[0]
        self.K = user_node.shape[1]
        


        user_node_update = torch.zeros((batch,self.K,self.ch+self.in_dim))

        mean_user = element_wise_mean(user_node)
        
        for k in range(self.K):
            sig_array = torch.zeros((batch,1)).to('cuda')

            if k != self.K:
                max_user = element_wise_max(torch.cat((user_node[:,:k,:],user_node[:,k+1:,:]),dim=1))
            else:
                max_user = element_wise_max(user_node[:,:k,:])
            
            
            temp_update_user = self.f2_user(torch.cat((user_node[:,k,:],max_user,RIS_node),dim=1))
            
            user_node_update[:,k,:] = torch.cat((user_node[:,k,:],temp_update_user),dim=1)
     
        temp_update_RIS = self.f3_RIS(torch.cat((RIS_node,mean_user),dim=1))
        RIS_node_update = torch.cat((RIS_node,temp_update_RIS),dim=1)

        

        return user_node_update, RIS_node_update




class readout_BS(nn.Module):
    def __init__(self,in_dim,M,Pt):
        super(readout_BS,self).__init__()
        self.M = M
        self.in_dim = in_dim
        self.Pt = Pt
        self.L = nn.Linear(self.in_dim,2*self.M)

    def forward(self,x):
        W = self.L(x)
        W = torch.transpose(W,2,1)
        K = W.shape[2]
        W = F.normalize(W,dim=2)*np.sqrt(self.Pt)/K

        return W, x

class readout_RIS(nn.Module):
    def __init__(self,in_dim,N,b):
        super(readout_RIS,self).__init__()
        self.in_dim = in_dim
        self.N = N
        self.b = b
        self.classifier = nn.Linear(self.in_dim,2*self.N)
        
    def forward(self,x):
        output = self.classifier(x)
        phase_re = output[:,:self.N].unsqueeze(2)
        phase_im = output[:,self.N:].unsqueeze(2)
        phase = torch.cat((phase_re,phase_im),dim=2)
        output = F.normalize(phase,dim=2)

        return output, x
        



class node_update(nn.Module):
    def __init__(self,in_dim,M,N,L,D,Pt,b):
        super(node_update,self).__init__()
        self.in_dim_user = in_dim*3
        self.in_dim_RIS = in_dim*2
        self.in_dim = in_dim
        self.N = N
        self.D = D
        self.M = M
        self.L = L
        self.Pt = Pt
        self.b = b
        self.init_user = user_init(in_dim,self.M,self.N,in_dim)
        self.init_RIS = RIS_init(in_dim,self.M,self.N,self.L,in_dim)
        update_list = []
        for d in range(self.D):
            update_list.append(node_update_layer(in_dim*(d+1),self.M,self.N,in_dim))
        self.update_list = nn.ModuleList(update_list)
        self.readout_BS = readout_BS(in_dim*(self.D+1),self.M,self.Pt)
        self.readout_RIS = readout_RIS(in_dim*(self.D+1),self.N,self.b)

        self.phase2_RIS_init = phase2_RIS_init(in_dim*(self.D+1), M, N, L, in_dim)
        self.phase2_user_init = phase2_user_init(in_dim, M, N, L, in_dim)

        phase2_list = []
        for d in range(self.D):
            phase2_list.append(node_update_layer(in_dim*(d+1)+in_dim*(self.D+1),self.L,self.N,in_dim))
        self.phase2_list = nn.ModuleList(phase2_list)
        self.readout_relay = readout_BS(in_dim*(self.D+1)*2,self.L,self.Pt)
        self.readout_RIS_phase2 = readout_RIS(in_dim*(self.D+1)*2,self.N,self.b)




    def forward(self,RIS_node, user_node, RIS_node_1, user_node_1, H, C, channel_relay_user, channel_bs_user, relay_channel):
        batch_size = RIS_node.shape[0]
        self.K = user_node.shape[1]
        relay_channel = torch.Tensor(np.concatenate((relay_channel.reshape((batch_size,-1)).real,relay_channel.reshape((batch_size,-1)).imag),axis=1)).to('cuda')
        channel_relay_user = torch.Tensor(np.concatenate((channel_relay_user.reshape((batch_size,1,-1,self.K)).real,channel_relay_user.reshape((batch_size,1,-1,self.K)).imag),axis=1)).to('cuda')
        channel_bs_user = torch.Tensor(np.concatenate((channel_bs_user.reshape((batch_size,1,-1,self.K)).real,channel_bs_user.reshape((batch_size,1,-1,self.K)).imag),axis=1)).to('cuda')



        z_user = torch.zeros((user_node.shape[0],self.K,self.in_dim)).to('cuda')
        for k in range(self.K):
            z_user[:,k,:] = self.init_user(user_node[:,k,:,:,:],channel_bs_user[:,:,:,k])
        user_node = z_user

        z_RIS = self.init_RIS(RIS_node,relay_channel,channel_bs_user).to('cuda')
        RIS_node = z_RIS

        for i, _ in enumerate(self.update_list):
            update_layer = self.update_list[i]
            user_node, RIS_node = update_layer(user_node, RIS_node, H, channel_bs_user)
            user_node = user_node.to('cuda')
            RIS_node = RIS_node.to('cuda')

        W, prev_feature_user = self.readout_BS(user_node)
        Theta, prev_feature_RIS = self.readout_RIS(RIS_node)

        

        
        z_user = torch.zeros((user_node_1.shape[0],self.K,self.in_dim*(self.D+2))).to('cuda')
        
        for k in range(self.K):
            z_user[:,k,:] = self.phase2_user_init(user_node_1[:,k,:,:,:],prev_feature_user[:,k,:],channel_relay_user[:,:,:,k])
            
        user_node = z_user
        

        z_RIS = self.phase2_RIS_init(RIS_node_1,prev_feature_RIS,channel_relay_user)
        RIS_node = z_RIS

        for i, _ in enumerate(self.phase2_list):
            update_layer = self.phase2_list[i]
            user_node, RIS_node = update_layer(user_node, RIS_node, C, channel_relay_user)
            user_node = user_node.to('cuda')
            RIS_node = RIS_node.to('cuda')

        Relay, _ = self.readout_relay(user_node)
        Theta_2, _ = self.readout_RIS_phase2(RIS_node)


        return W, Relay, Theta, Theta_2










class RIS_init(nn.Module):
    def __init__(self,in_dim,M,N,L,ch):
        super(RIS_init,self).__init__()
        self.in_dim = in_dim

        self.M = M
        self.L = L
        self.in_dim_R = 2*M*L
        self.ch = ch

        self.FE = nn.Sequential(
            nn.Linear(self.in_dim_R,self.ch),
            # nn.BatchNorm1d(self.ch),
            nn.ReLU(),
            nn.Linear(self.ch,self.ch//2)
        )

        self.C = nn.Sequential(
            nn.Conv2d(2*(N+1),ch,1),
            # nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch,ch//2,1),
        )

        self.out = nn.Linear(M*ch//4,ch//2)

    def forward(self,x,relay_channel,channel_bs_user):
        batch_size = x.shape[0]

        x = torch.cat((x,channel_bs_user),dim=1)

        output = self.C(x)
        output = torch.mean(output,dim=3)
        output = torch.mean(output,dim=2)
        # output = output.reshape((batch_size,-1))
        # output = self.out(output)
        
        relay_channel = relay_channel.reshape((batch_size,-1))
        feature = self.FE(relay_channel)
        output = torch.cat((output,feature),dim=1)
        return output


class user_init(nn.Module):
    def __init__(self,in_dim,M,N,ch):
        super(user_init,self).__init__()
        self.in_dim = in_dim
        self.M = M

        # self.C = nn.Sequential(
        #     nn.Conv2d(2,ch,1),
        #     nn.BatchNorm2d(ch),
        #     nn.ReLU(),
        #     nn.Conv2d(ch,1,1),
        #     nn.Flatten(),
        #     nn.Linear(M*(N+1),ch)
        # )

        self.C = nn.Sequential(
            nn.Linear(2*M*N+2*M,2*ch),
            # nn.BatchNorm1d(2*ch),
            nn.ReLU(),
            nn.Linear(2*ch,ch)
        )
    def forward(self,x,channel_bs_user):
        batch_size = x.shape[0]
        x1 = channel_bs_user.reshape((batch_size,-1))
        x = x.reshape((batch_size,-1))
        x = torch.cat((x,x1),dim=1)
        output = self.C(x)
        # output = torch.cat((output,x),dim=2)
        return output


class phase2_RIS_init(nn.Module):
    def __init__(self,in_dim,M,N,L,ch):
        super(phase2_RIS_init,self).__init__()
        self.in_dim = in_dim
        self.M = M
        self.L = L
        self.C = nn.Sequential(
            nn.Conv2d(2*(N+1),ch,1),
            # nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch,ch,1),
        )

        # self.out = nn.Linear(L*ch//4,ch)

        # self.C = nn.Linear(2*L*K*N,ch)
        self.L_layer = nn.Linear(self.in_dim+ch,ch)
    def forward(self,x,prev_feature,channel_relay_user):
        batch_size = x.shape[0]

        x = torch.cat((x,channel_relay_user),dim=1)


        output = self.C(x)
        output = torch.mean(output,dim=3)
        output = torch.mean(output,dim=2)
        # output = output.reshape((batch_size,-1))
        # output = self.out(output)
        output = torch.cat((output,prev_feature),dim=1)
        # output = self.L_layer(output)
        # output = torch.cat((output,x),dim=2)
        return output



class phase2_user_init(nn.Module):
    def __init__(self,in_dim,M,N,L,ch):
        super(phase2_user_init,self).__init__()
        self.in_dim = in_dim
        self.M = M
        self.N = N
        self.L = L

        # self.C = nn.Sequential(
        #     nn.Conv2d(2,ch,1),
        #     nn.BatchNorm2d(ch),
        #     nn.ReLU(),
        #     nn.Conv2d(ch,1,1),
        #     nn.Flatten(),
        #     nn.Linear(L*(N+1),ch)
        # )

        self.L_layer = nn.Linear(self.in_dim+2*self.M+2*self.L,self.in_dim)

        self.C = nn.Sequential(
            nn.Linear(2*L*N+2*L,2*ch),
            # nn.BatchNorm1d(2*ch),
            nn.ReLU(),
            nn.Linear(2*ch,ch)
        )
    def forward(self,x,prev_feature,channel_relay_user):
        
        batch_size = x.shape[0]
        x1 = channel_relay_user.reshape((batch_size,-1))
        x = x.reshape((batch_size,-1))
        x = torch.cat((x,x1),dim=1)

        output = self.C(x)

        output = torch.cat((output,prev_feature),dim=1)
        return output











