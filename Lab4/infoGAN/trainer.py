import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import pandas as pd

DEBUG = False
D_probs_real_list, D_probs_fake_list, G_probs_fake_list = [], [], []
G_loss_list, D_loss_list, Q_loss_list = [], [], []

class StableBCELoss(nn.modules.Module):
  def __init__(self):
    super(StableBCELoss, self).__init__()
  
  def forward(self, input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

class log_gaussian:

  def __call__(self, x, mu, var):

    logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - \
            (x-mu).pow(2).div(var.mul(2.0)+1e-6)
    
    return logli.sum(1).mean().mul(-1)

class Trainer:

  def __init__(self, G, FE, D, Q):

    self.G = G
    self.FE = FE
    self.D = D
    self.Q = Q

    self.batch_size = 1#00

  def _noise_sample(self, dis_c, con_c, noise, bs):

    idx = np.random.randint(10, size=bs)
    c = np.zeros((bs, 10))
    c[range(bs),idx] = 1.0

    dis_c.data.copy_(torch.Tensor(c))
    con_c.data.uniform_(-1.0, 1.0)
    noise.data.uniform_(-1.0, 1.0)

    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 64, 1, 1)

    return z, idx, Variable(torch.from_numpy(c.astype('float32')).squeeze())

  def train(self):

    real_x = torch.FloatTensor(self.batch_size, 1, 28, 28).cuda()
    label = torch.FloatTensor(self.batch_size).cuda()
    dis_c = torch.FloatTensor(self.batch_size, 10).cuda()
    con_c = torch.FloatTensor(self.batch_size, 2).cuda()
    noise = torch.FloatTensor(self.batch_size, 62).cuda()

    real_x = Variable(real_x)
    label = Variable(label, requires_grad=False)
    dis_c = Variable(dis_c)
    con_c = Variable(con_c)
    noise = Variable(noise)

    #criterionD = nn.BCELoss().cuda()
    criterionD = StableBCELoss().cuda()
    criterionQ_dis = nn.CrossEntropyLoss().cuda()
    criterionQ_con = nn.MSELoss()#log_gaussian()

    optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

    T = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    dataset = dset.MNIST('./dataset', transform=T, download=True)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

    # fixed random variables
    c = np.linspace(-1, 1, 10).reshape(1, -1)
    c = np.repeat(c, 10, 0).reshape(-1, 1)

    c1 = np.hstack([c, np.zeros_like(c)])
    c2 = np.hstack([np.zeros_like(c), c])

    idx = np.arange(10).repeat(10)
    one_hot = np.zeros((100, 10))
    one_hot[range(100), idx] = 1
    fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)


    for epoch in range(100):
      for num_iters, batch_data in enumerate(dataloader, 0):

        # real part
        optimD.zero_grad()
        
        x, _ = batch_data

        bs = x.size(0)
        real_x.data.resize_(x.size())
        label.data.resize_(bs)
        dis_c.data.resize_(bs, 10)
        con_c.data.resize_(bs, 2)
        noise.data.resize_(bs, 52)
        
	real_x.data.copy_(x)        
	if(DEBUG):
            print("x.shape:" + str(x.shape))

        fe_out1 = self.FE(real_x)
        if(DEBUG):
            print("fe_out1.shape:" + str(fe_out1.shape))

        D_probs_real = self.D(fe_out1)
        label.data.fill_(1)	
        if(DEBUG):
            print("probs_real:" + str(D_probs_real.shape))
            #print(probs_real)
            print("label:" + str(label.shape))
            #print(label)

        loss_real = criterionD(D_probs_real, label)
        loss_real.backward()

        # fake part
        z, idx, c = self._noise_sample(dis_c, con_c, noise, bs)
	if(DEBUG):
            print("z.shape:" + str(z.shape))

        fake_x = self.G(z)
        if(DEBUG):
            print("fake_x:" + str(fake_x.shape))

        fe_out2 = self.FE(fake_x.detach())
        D_probs_fake = self.D(fe_out2)
        label.data.fill_(0)
        loss_fake = criterionD(D_probs_fake, label)
        loss_fake.backward()

        D_loss = loss_real + loss_fake

        optimD.step()
        
        # G and Q part
        optimG.zero_grad()

        fe_out = self.FE(fake_x)
        G_probs_fake = self.D(fe_out)
        label.data.fill_(1.0)

        reconstruct_loss = criterionD(G_probs_fake, label)
	if(DEBUG):
            print(-torch.mean(torch.log(G_probs_fake + 1e-8)))
	    print(reconstruct_loss)
       
        if(DEBUG):
            print("before: fe_out:" + str(fe_out.shape))
	fe_out = fe_out.view(fe_out.shape[0], 8192)
        if(DEBUG):
            print("after: fe_out:" + str(fe_out.shape))

        #q_logits, q_mu, q_var = self.Q(fe_out)
        q_var = self.Q(fe_out)
	q_logits = q_var
        class_ = torch.LongTensor(idx).cuda()
        target = Variable(class_)
	'''
	print(q_var.shape)
	print(q_logits)
	raw_input('')
        dis_loss = criterionQ_dis(q_logits, target)
        con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1
        '''
	'''
	print('torch.log(c + 1e-8):')
	print(torch.log(c + 1e-8))
	raw_input('')
	print('-torch.sum(c * torch.log(c + 1e-8)')
	print(-torch.sum(c * torch.log(c + 1e-8), dim=1))
	raw_input('')
	print('torch.mean(-torch.sum(c * torch.log(c + 1e-8), dim=1)')
	print(torch.mean(-torch.sum(c * torch.log(c + 1e-8), dim=1)))
	raw_input('')
	crossent_loss = torch.mean(-torch.sum(c * torch.log(q_var.cpu().data + 1e-8), dim=1))
	ent_loss = torch.mean(-torch.sum(c * torch.log(c + 1e-8), dim=1))
	'''

	# corss entropy
	digit_classify_loss = criterionQ_dis(q_var[:10], target)
	if(DEBUG):
		print(digit_classify_loss)
	
        conti_loss = criterionQ_con(q_var[:, 10:], z[:,-2:].view(bs,2))
	#conti_loss = criterionQ_con(pred_c[:,8:], z[:,-2:].view(bs,2))
	if(DEBUG):
		print(conti_loss)

        G_loss = reconstruct_loss + digit_classify_loss + conti_loss
        G_loss.backward()
        optimG.step()
	#raw_input("iter end.")

        if num_iters % 500 == 0:

          print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
            epoch, num_iters, D_loss.data.cpu().numpy(),
            G_loss.data.cpu().numpy())
          )

	  '''
          noise.data.copy_(fix_noise)
          dis_c.data.copy_(torch.Tensor(one_hot))

          con_c.data.copy_(torch.from_numpy(c1))
          z = torch.cat([noise, dis_c, con_c], 1).view(-1, 64, 1, 1)
          x_save = self.G(z)
          save_image(x_save.data, './tmp/c1.png', nrow=10)

          con_c.data.copy_(torch.from_numpy(c2))
          z = torch.cat([noise, dis_c, con_c], 1).view(-1, 64, 1, 1)
          x_save = self.G(z)
          save_image(x_save.data, './tmp/c2.png', nrow=10)
	  '''

	  G_loss_list.append(G_loss.data.cpu().numpy())
          D_loss_list.append(D_loss.data.cpu().numpy())
          Q_loss_list.append(digit_classify_loss.data.cpu().numpy() + conti_loss.data.cpu().numpy())
          
          df1 = pd.DataFrame([], columns=["generator","discriminator","Qdis","Qcon"])
          df1['G_loss'] = G_loss_list
          df1["D_loss"] = D_loss_list
          df1['Q_loss'] = Q_loss_list
          
          df1=df1.astype(float)
          plot1 = df1.plot()
          fig1 = plot1.get_figure()
          #fig1.savefig("loss.png")
          df1.to_csv('./results/loss.csv', index=False)
          
          D_probs_real_list.append(D_probs_real.mean().data.cpu().numpy())
          D_probs_fake_list.append(D_probs_fake.mean().data.cpu().numpy())
          G_probs_fake_list.append(G_probs_fake.mean().data.cpu().numpy())


          df2 = pd.DataFrame([], columns=["probs_real","probs_fake_before","probs_fake_after"])
          df2['probs_real'] = D_probs_real_list
          df2["probs_fake_before"] = D_probs_fake_list
          df2['probs_fake_after'] = G_probs_fake_list
          df2.to_csv('./results/probs.csv', index=False)
          df2=df2.astype(float)
          plot2 = df2.plot()
          fig2 = plot2.get_figure()
          #fig2.savefig("prob.png")


	  #device = torch.device("cuda:0")
          #fixed_noise = torch.randn(64, 64, 1, 1, device=device)
          
          #vutils.save_image(real_x.detach(), './results/real_samples.png', normalize=True)
          #fake = self.G(fixed_noise)
          
          #vutils.save_image(fake.detach(), './results/fake_samples_epoch_%03d.png' % (epoch), normalize=True)

	  torch.save(self.FE.state_dict(), './FE.pt')
	  torch.save(self.G.state_dict(), './G.pt')
	  torch.save(self.D.state_dict(), './D.pt')
	  torch.save(self.Q.state_dict(), './Q.pt')

	  raw_input('')
