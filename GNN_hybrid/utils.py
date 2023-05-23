import numpy as np
import torch

def generate_location(K,center):
    locations = np.zeros((K,2))
    radius = 10
    
    for k in range(K):
        angle = 2*np.pi*np.random.uniform(0,1)
        rad = radius * np.random.uniform(0,1)
        locations[k,:] = center + np.array([rad*np.cos(angle), rad*np.sin(angle)])
    return locations



def element_wise_mean(x):
    batch = x.shape[0]
    return torch.mean(x,dim=1)

def element_wise_max(x):
    return torch.amax(x,dim=1)

def cal_loss(W, Relay, Theta, Theta_2, H, C, channel_relay_user, channel_bs_user, relay_channel,threshold,relay_SINR_channel):
    M = W.shape[1]//2
    L = Relay.shape[1]//2
    W_re = W[:,:M,:]
    W_im = W[:,M:,:]
    phase_re = Theta[:,:,0]
    phase_im = Theta[:,:,1]
    cas_channel = H
    K = Relay.shape[2]
    num_sample = H.shape[0]
    rate = []
    sum_rate = torch.zeros(num_sample).to('cuda')
    phase_T = torch.cat((phase_re,phase_im),dim=1).unsqueeze(2)
    sigma_2 = (2e-5)**2
    
    gamma_1 = []

    for k1 in range(K):
        A_tmp_tran = np.transpose(cas_channel[:,:,:,k1],(0,2,1))
        A_tmp_real1 = np.concatenate([A_tmp_tran.real,A_tmp_tran.imag],axis=2)
        A_tmp_real2 = np.concatenate([-A_tmp_tran.imag,A_tmp_tran.real],axis=2)
        A_tmp_real = np.concatenate([A_tmp_real1,A_tmp_real2],axis=1)
        A_T_k = torch.Tensor(A_tmp_real).to('cuda')

        h_bs = torch.cat((torch.Tensor(channel_bs_user[:,k1,:].real),torch.Tensor(channel_bs_user[:,k1,:].imag)),dim=1).unsqueeze(2).to('cuda')
        phase_A_T_k = torch.bmm(A_T_k.transpose(2,1),phase_T) + h_bs

        signal_power = []
        sum_power = torch.zeros(W_re.shape[0]).to('cuda')
        for k2 in range(K):
            W_real = W_re[:,:,k2].unsqueeze(2)
            W_imag = W_im[:,:,k2].unsqueeze(2)
            W_mat1 = torch.cat((W_real,W_imag),dim=2)
            W_mat2 = torch.cat((-W_imag,W_real),dim=2)
            W_mat = torch.cat((W_mat1,W_mat2),dim=1)

            z = torch.bmm(W_mat.transpose(2,1),phase_A_T_k).squeeze()
            z = torch.square(z[:,0]) + torch.square(z[:,1])
            signal_power.append(z)
            sum_power += z
        
        gamma_k_1 = signal_power[k1]/(sum_power-signal_power[k1]+sigma_2)
        gamma_1.append(gamma_k_1)

    W_re = Relay[:,:L,:]
    W_im = Relay[:,L:,:]
    phase_re = Theta_2[:,:,0]
    phase_im = Theta_2[:,:,1]
    cas_channel = C
    rate = []
    phase_T = torch.cat((phase_re,phase_im),dim=1).unsqueeze(2)
    sigma_2 = (2e-5)**2
    
    gamma_2 = []

    for k1 in range(K):
        A_tmp_tran = np.transpose(cas_channel[:,:,:,k1],(0,2,1))
        A_tmp_real1 = np.concatenate([A_tmp_tran.real,A_tmp_tran.imag],axis=2)
        A_tmp_real2 = np.concatenate([-A_tmp_tran.imag,A_tmp_tran.real],axis=2)
        A_tmp_real = np.concatenate([A_tmp_real1,A_tmp_real2],axis=1)
        A_T_k = torch.Tensor(A_tmp_real).to('cuda')

        h_relay = torch.cat((torch.Tensor(channel_relay_user[:,k1,:].real),torch.Tensor(channel_relay_user[:,k1,:].imag)),dim=1).unsqueeze(2).to('cuda')
        phase_A_T_k = torch.bmm(A_T_k.transpose(2,1),phase_T) + h_relay

        signal_power = []
        sum_power = torch.zeros(W_re.shape[0]).to('cuda')
        for k2 in range(K):
            W_real = W_re[:,:,k2].unsqueeze(2)
            W_imag = W_im[:,:,k2].unsqueeze(2)
            W_mat1 = torch.cat((W_real,W_imag),dim=2)
            W_mat2 = torch.cat((-W_imag,W_real),dim=2)
            W_mat = torch.cat((W_mat1,W_mat2),dim=1)

            z = torch.bmm(W_mat.transpose(2,1),phase_A_T_k).squeeze()
            z = torch.square(z[:,0]) + torch.square(z[:,1])
            signal_power.append(z)
            sum_power += z
        
        gamma_k_2 = signal_power[k1]/(sum_power-signal_power[k1]+sigma_2)
        gamma_2.append(gamma_k_2)

    SINR_1 = []
    SINR_2 = []
    for k in range(K):
        rate_k = torch.log2(1+gamma_1[k]+gamma_2[k])
        SINR_1.append(torch.mean(gamma_1[k]).item())
        SINR_2.append(torch.mean(gamma_2[k]).item())
        
        if k==0:
            sum_rate = rate_k
            
        else:
            sum_rate += rate_k
            

    channel_bs_ris,channel_relay_ris,channel_bs_relay = relay_SINR_channel
    phase_re = Theta[:,:,0]
    phase_im = Theta[:,:,1]
    channel_relay_ris_H = im2re(np.conj(channel_relay_ris.transpose((0,2,1))))
    channel_bs_ris = im2re(channel_bs_ris)
    channel_bs_relay = im2re(channel_bs_relay)
    phase_re = torch.diag_embed(phase_re)
    phase_im = torch.diag_embed(phase_im)
    phase_1 = torch.cat((phase_re,-phase_im),dim=2)
    phase_2 = torch.cat((phase_im,phase_re),dim=2)
    phase = torch.cat((phase_1,phase_2),dim=1).cpu()
    H_prime = torch.matmul(channel_relay_ris_H,torch.matmul(phase,channel_bs_ris)) + channel_bs_relay
    alpha = torch.zeros((num_sample,H_prime.shape[1],K))

    for k in range(K):
        alpha[:,:,k] = torch.matmul(H_prime,W[:,:,k].cpu().reshape((num_sample,-1,1))).squeeze()
    
    SINR = torch.zeros((num_sample,K))
    for k in range(K):
        signal = torch.pow(torch.norm(alpha[:,:,k],dim=1),4)
        interference = torch.zeros(num_sample)

        for k1 in range(K):
            if k1 == k:
                continue
            else:
                a_k = alpha[:,:L,k].detach().numpy() + 1j * alpha[:,L:,k].detach().numpy()
                a_k = a_k.reshape((num_sample,1,L))
                a_k = np.matmul(np.conj(a_k.transpose((0,2,1))),a_k)
                a_j = alpha[:,:L,k1].detach().numpy() + 1j * alpha[:,L:,k1].detach().numpy()
                a_j = a_j.reshape((num_sample,1,L))
                inter = np.matmul(np.conj(a_j),np.matmul(a_k,a_j.transpose((0,2,1))))
                interference = interference + torch.Tensor(inter.real).squeeze()
        
        interference += sigma_2 * torch.pow(torch.norm(alpha[:,:,k],dim=1),2)
        SINR[:,k] = signal / interference

    
    SINR = SINR - threshold
    num = torch.sum(SINR>0)
    zero = torch.zeros(SINR.shape)
    penalty = torch.minimum(SINR,zero)
    beta = 1000

    loss = - torch.sum(sum_rate)/num_sample - torch.sum(penalty)/num_sample*beta
    

    return loss, torch.sum(sum_rate)/num_sample, num/num_sample/K, SINR_1, SINR_2

def im2re(M):
    M1 = np.concatenate((M.real,-M.imag),axis=2)
    M2 = np.concatenate((M.imag,M.real),axis=2)
    M_mat = np.concatenate((M1,M2),axis=1)
    M_mat = torch.Tensor(M_mat)

    return M_mat


def gen_LOS(num_rev,num_trans,Rician_factor):
    AoA = np.ones((num_rev,1),dtype=np.complex128)
    AoD = np.ones((num_trans,1),dtype=np.complex128)
    angle_AoA = 2*np.pi*np.random.uniform(0,1)
    for n in range(1,num_rev):
        AoA[n,:] = np.exp(1j*n*np.pi*np.sin(angle_AoA))
    angle_AoD = 2*np.pi*np.random.uniform(0,1)
    for n in range(1,num_trans):
        AoD[n,:] = np.exp(1j*n*np.pi*np.sin(angle_AoD))
    mat_LOS = np.dot(AoA,np.conj(AoD.T))
    LOS = np.sqrt(Rician_factor/(Rician_factor+1))*mat_LOS

    return LOS



class Channel():
    def __init__(self,num_trans,num_rev,factor):
        self.num_trans = num_trans
        self.num_rev = num_rev
        self.factor = factor
        self.AoA = np.ones((num_rev,1),dtype=np.complex128)
        self.AoD = np.ones((num_trans,1),dtype=np.complex128)
        self.mat = np.zeros((num_rev,num_trans),dtype=np.complex128)
        self.NLOS = np.zeros((num_rev,num_trans),dtype=np.complex128)
        self.LOS = np.zeros((num_rev,num_trans),dtype=np.complex128)

    def generate_value(self,mean_NLOS,cov_NLOS,LOS):
        self.LOS = LOS
        R_NLOS = np.linalg.cholesky(cov_NLOS)
        mat_real_NLOS = np.ones((self.num_rev,self.num_trans))*mean_NLOS + np.matmul(np.random.randn(self.num_rev,self.num_trans),R_NLOS)
        mat_imag_NLOS = np.ones((self.num_rev,self.num_trans))*mean_NLOS + np.matmul(np.random.randn(self.num_rev,self.num_trans),R_NLOS)
        self.NLOS = np.sqrt(1/(self.factor+1))*(mat_real_NLOS+1j*mat_imag_NLOS)/np.sqrt(2)
        
        self.mat = self.NLOS + self.LOS



    def large_scale_loss(self,fading_NLOS,exp_NLOS,fading_LOS,exp_LOS,dist):
        self.large_scale_fading_NLOS = fading_NLOS*dist**(-exp_NLOS)
        self.large_scale_fading_LOS = fading_LOS*dist**(-exp_LOS)
        self.ori_mat = self.large_scale_fading_NLOS*self.NLOS+self.large_scale_fading_LOS*self.LOS
        self.mat = self.large_scale_fading_NLOS*self.NLOS+self.large_scale_fading_LOS*self.LOS

        return self.mat

    def time_varying(self,prev_NLOS,prev_LOS):
        epsilon = 0.01
        self.NLOS = prev_NLOS*np.sqrt(1-epsilon**2) + self.NLOS*epsilon
        self.LOS = prev_LOS*np.sqrt(1-epsilon**2) + self.LOS*epsilon




def generate_channel(M,N,L,K,batch_size,LOS_bs_ris,LOS_bs_relay,LOS_relay_ris,sigma=0):
    loc_RIS = np.array([100,50])
    loc_relay = np.array([100,-50])
    loc_BS = np.array([0,0])
    user_center = np.array([200,0])
    
    
    Rician_factor = 10
    background_noise = 2e-12
    fading_BS_users = 10**(-4.5)
    path_loss_exp_BS_user = 3.5
    fading_BS_relay = 10**(-4.5)
    fading_relay_users = 10**(-4.5)
    fading_BS_RIS = 10**(-0.5)
    fading_RIS_users = 10**(-0.5)
    fading_relay_RIS = 10**(-0.5)
    path_loss_exp_LOS = 2
    path_loss_exp_NLOS = 2.5
    channel_bs_ris = []
    channel_relay_ris = []
    channel_ris_user = []
    channel_bs_relay = []
    channel_relay_user = []
    channel_bs_user = []
    theta = []

    epsilon = 0.01

    index_n = np.arange(N)
    DFT_matrix = np.diag(np.exp(-2j*np.pi*index_n/N))

    
    h_LOS_relay = gen_LOS(1,L,0)
    h_LOS_BS = gen_LOS(1,M,0)

    for sample in range(batch_size):
        loc_user = generate_location(K,user_center)
        
        H_BS_RIS = Channel(M,N,Rician_factor)
        H_BS_RIS.generate_value(0,np.eye(M),LOS_bs_ris)
        H_BS_RIS = H_BS_RIS.large_scale_loss(fading_BS_RIS,path_loss_exp_NLOS,fading_BS_RIS,path_loss_exp_LOS,np.linalg.norm(loc_RIS))
        channel_bs_ris.append(H_BS_RIS)

        H_BS_relay = Channel(M,L,0)
        H_BS_relay.generate_value(0,np.eye(M),LOS_bs_relay)
        H_BS_relay = H_BS_relay.large_scale_loss(fading_BS_relay,path_loss_exp_NLOS,0,0,np.linalg.norm(loc_relay))
        channel_bs_relay.append(H_BS_relay)

        H_relay_RIS = Channel(L,N,Rician_factor)
        H_relay_RIS.generate_value(0,np.eye(L),LOS_relay_ris)
        H_relay_RIS = H_relay_RIS.large_scale_loss(fading_relay_RIS,path_loss_exp_NLOS,fading_relay_RIS,path_loss_exp_LOS,np.linalg.norm(loc_RIS-loc_relay))
        channel_relay_ris.append(H_relay_RIS)

        theta.append(DFT_matrix)

        tmp_ris_user = []
        tmp_relay_user = []
        tmp_bs_user =  []

        

        for k in range(K):
            h_LOS_RIS = gen_LOS(1,N,Rician_factor)
            h_RIS = Channel(N,1,Rician_factor)
            h_RIS.generate_value(0,np.eye(N),h_LOS_RIS)
            h_RIS = h_RIS.large_scale_loss(fading_RIS_users,path_loss_exp_NLOS,fading_RIS_users,path_loss_exp_LOS,np.linalg.norm(loc_user[k,:]-loc_RIS))
            tmp_ris_user.append(np.diag(h_RIS[0]))
        
        
            h_relay = Channel(L,1,0)
            h_relay.generate_value(0,np.eye(L),h_LOS_relay)
            h_relay = h_relay.large_scale_loss(fading_relay_users,path_loss_exp_NLOS,fading_relay_users,path_loss_exp_LOS,np.linalg.norm(loc_user[k,:]-loc_relay))
            tmp_relay_user.append(h_relay[0])

            h_bs = Channel(M,1,0)
            h_bs.generate_value(0, np.eye(M), h_LOS_BS)
            h_bs = h_bs.large_scale_loss(fading_BS_users,path_loss_exp_BS_user,0,0,np.linalg.norm(loc_user[k,:]))
            tmp_bs_user.append(h_bs[0])

        channel_ris_user.append(tmp_ris_user)
        channel_relay_user.append(tmp_relay_user)
        channel_bs_user.append(tmp_bs_user)
    scale = -7
    background_noise = background_noise/10**(scale)

    channel_bs_ris = np.array(channel_bs_ris)/np.sqrt(10**scale)
    channel_ris_user = np.array(channel_ris_user)/np.sqrt(10**scale)
    channel_bs_relay = np.array(channel_bs_relay)/10**scale
    channel_relay_ris = np.array(channel_relay_ris)/np.sqrt(10**scale)
    channel_relay_user = np.array(channel_relay_user)/10**scale
    channel_bs_user = np.array(channel_bs_user)/10**scale
    if sigma>0:
        perfect_channel_ris_user = np.copy(channel_ris_user)
        re_imperfect = np.random.normal(0,sigma,size=channel_ris_user.shape)
        im_imperfect = np.random.normal(0,sigma,size=channel_ris_user.shape)
        imperfect = (re_imperfect+1j*im_imperfect)/np.sqrt(2)
        channel_ris_user = channel_ris_user + imperfect

        perfect_channel_relay_user = np.copy(channel_relay_user)
        re_imperfect = np.random.normal(0,sigma,size=channel_relay_user.shape)
        im_imperfect = np.random.normal(0,sigma,size=channel_relay_user.shape)
        imperfect = (re_imperfect+1j*im_imperfect)/np.sqrt(2)
        channel_relay_user = channel_relay_user + imperfect

    combined_channel_H = np.zeros([batch_size,M,N,K],dtype=np.complex128)
    combined_channel_C = np.zeros([batch_size,L,N,K],dtype=np.complex128)

    for k in range(K):
        channel_ris_user_k = channel_ris_user[:,k,:,:]
        combined_ris = np.matmul(channel_bs_ris.transpose(0,2,1), channel_ris_user_k)
        combined_channel_H[:,:,:,k] = combined_ris

        combined_relay = np.matmul(channel_relay_ris.transpose(0,2,1), channel_ris_user_k)
        combined_channel_C[:,:,:,k] = combined_relay
    
    relay_channel = np.matmul(np.conj(channel_relay_ris.transpose(0,2,1)),np.matmul(theta,channel_bs_ris)) + channel_bs_relay

    relay_SINR_channel = channel_bs_ris,channel_relay_ris,channel_bs_relay

    if sigma>0:
        perfect_H = np.zeros([batch_size,M,N,K],dtype=np.complex128)
        perfect_C = np.zeros([batch_size,L,N,K],dtype=np.complex128)

        for k in range(K):
            channel_ris_user_k = perfect_channel_ris_user[:,k,:,:]
            combined_ris = np.matmul(channel_bs_ris.transpose(0,2,1), channel_ris_user_k)
            perfect_H[:,:,:,k] = combined_ris

            combined_relay = np.matmul(channel_relay_ris.transpose(0,2,1), channel_ris_user_k)
            perfect_C[:,:,:,k] = combined_relay

        return combined_channel_H, combined_channel_C, channel_relay_user, channel_bs_user, relay_channel, relay_SINR_channel, perfect_channel_ris_user, perfect_channel_relay_user, perfect_H, perfect_C  

    else:
        return combined_channel_H, combined_channel_C, channel_relay_user, channel_bs_user, relay_channel, relay_SINR_channel

def discrete_mapping(theta,num_bits):
    level = 2**num_bits
    phase_re =  torch.real(torch.exp(1j*torch.arange(level)/level*2*np.pi))
    phase_im =  torch.imag(torch.exp(1j*torch.arange(level)/level*2*np.pi))
    phase = torch.cat((phase_re.unsqueeze(1),phase_im.unsqueeze(1)),dim=1).to('cuda')
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            temp_theta = theta[i,j,:]
            temp_theta = temp_theta - phase
            temp_theta = torch.norm(temp_theta,dim=1)
            index = torch.argmin(temp_theta)
            theta[i,j,:] = phase[index]

    return theta





if __name__ == '__main__':
    num_test = 64
    M, N, L ,K = 8, 50, 4, 4
    Rician_factor = 10

    theta = torch.randn(64,50,2)
    discrete_mapping(theta,2)

    # W_re = torch.randn(64,M,K)
    # W_im = torch.randn(64,M,K)
    # phase_re = torch.randn(64,N)
    # phase_im = torch.randn(64,N)

    # loss = cal_loss(W_re,W_im,phase_re,phase_im,cas_channel)

    # print(loss)
