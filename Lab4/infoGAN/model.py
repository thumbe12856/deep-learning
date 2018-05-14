import torch.nn as nn


class FrontEnd(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self):
    super(FrontEnd, self).__init__()

    '''
    self.main = nn.Sequential(
      nn.Conv2d(1, 64, 4, 2, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 7, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True),
    )
    '''

    self.main = nn.Sequential(
      nn.Conv2d(1, 64, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(128, 256, 4, 2, 1, bias=False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(256, 512, 4, 2, 1, bias=False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2, inplace=True),
    )

  def forward(self, x):
    output = self.main(x)
    return output


class D(nn.Module):

  def __init__(self):
    super(D, self).__init__()
    
    self.main = nn.Sequential(
      #nn.Conv2d(1024, 1, 1),
      nn.Conv2d(512, 1, kernel_size=4, stride=1, bias=False),
      nn.Sigmoid()
    )
    

  def forward(self, x):
    output = self.main(x).view(1)
    return output


class Q(nn.Module):

  def __init__(self):
    super(Q, self).__init__()

    '''
    self.conv = nn.Conv2d(1024, 128, 1, bias=False)
    self.bn = nn.BatchNorm2d(128)
    self.lReLU = nn.LeakyReLU(0.1, inplace=True)
    self.conv_disc = nn.Conv2d(128, 10, 1)
    self.conv_mu = nn.Conv2d(128, 2, 1)
    self.conv_var = nn.Conv2d(128, 2, 1)
    '''
    self.conv = nn.Sequential(
      nn.Linear(8192, out_features=100, bias=True),
      nn.ReLU(True),
      nn.Linear(100, out_features=12, bias=True),
    )

  def forward(self, x):
    """'''
    disc_logits = self.conv_disc(y).squeeze()

    mu = self.conv_mu(y).squeeze()
    var = self.conv_var(y).squeeze().exp()

    return disc_logits, mu, var 
    '''
    """
    y = self.conv(x)
    sm = nn.Softmax()
    return y


class G(nn.Module):

  def __init__(self):
    super(G, self).__init__()

    self.main = nn.Sequential(

      #nn.ConvTranspose2d(74, 1024, 1, 1, bias=False),
      nn.ConvTranspose2d(64, 512, kernel_size=4, stride=1, bias=False),
      #nn.BatchNorm2d(1024),
      nn.BatchNorm2d(512),
      nn.ReLU(True),

      #nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
      nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
      #nn.BatchNorm2d(128),
      nn.BatchNorm2d(256),
      nn.ReLU(True),

      #nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
      #nn.BatchNorm2d(64),
      nn.BatchNorm2d(128),
      nn.ReLU(True),

      #nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),

      nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
      nn.Tanh()
    )

  def forward(self, x):
    output = self.main(x)
    return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
