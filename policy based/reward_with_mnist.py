{\rtf1\ansi\ansicpg949\cocoartf2513
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset129 AppleSDGothicNeo-Regular;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import torch\
import torch.nn as nn\
import torch.optim as optim\
from torchvision import datasets, transforms\
\
\
train_dataset = datasets.MNIST('.../mnist_data/', download=False)\
\
'''\
dir(train_dataset)\
'''\
train_data = train_dataset.train_data\
train_label = train_dataset.train_labels\
\
'''\
train_data\
train_label\
'''\
\
'''\
make target -> predict net\
'''\
\
class RNetwork(nn.Module):\
    def __init__(self, n_input):\
        super(RNetwork, self).__init__()\
        self.pred = nn.Sequential(\
            nn.Linear(n_input, 256),\
            nn.ReLU(),\
            nn.Linear(256, 512),\
            nn.ReLU(),\
            nn.Linear(512, 128),\
            nn.ReLU(),\
            nn.Linear(128,1)\
        )\
        self.target = nn.Sequential(\
            nn.Linear(n_input, 256),\
            nn.ReLU(),\
            nn.Linear(256, 512),\
            nn.ReLU(),\
            nn.Linear(512, 128),\
            nn.ReLU(),\
            nn.Linear(128, 1)\
        )\
        self.target.apply(self.random_init)\
\
    def random_init(self, m): # for target_network\
        if type(m) == nn.Linear:\
            nn.init.orthogonal_(m.weight)\
\
    def forward(self,x):\
        pred = self.pred(x)\
        target = self.target(x)\
        return pred, target\
\
\
\
def train(pred, target, opt):\
    mse = nn.MSELoss()\
    loss = mse(target.detach(),pred)\
    opt.zero_grad()\
    loss.backward()\
    opt.step()\
    return torch.mean(loss) / pred.shape[0]\
\
\
'''\
test with some data from mnist\
'''\
epoch = 10\
\
net = RNetwork(28*28)\
\
test_data = train_data[:100,].reshape(100,-1).float()\
new_data = train_data[100:105,].reshape(5,-1).float()\
opt = optim.Adam(net.pred.parameters(),lr=0.0005)\
pred, target = net(new_data)\
\
for i in range(epoch):\
    loss = train(pred, target, opt)\
    print('epi : %d Loss: %f' %(i,loss))\
    pred, target = net(test_data)\
\
pred, target = net(new_data)\
loss = train(pred, target, opt)\
print('epi : Extra Loss: %f' %(loss))\
\
\
\
'''\
\'bb\'f5\'b7\'ce\'bf\'ee \'b5\'a5\'c0\'cc\'c5\'cd\'b0\'a1 \'b5\'e9\'be\'ee\'b0\'a1\'b8\'e9 LOSS\'b0\'a1 \'b4\'c3\'be\'ee\'b3\'b2\'c0\'bb \'c8\'ae\'c0\'ce -> exploration\'c0\'bb \'c8\'ae\'b4\'eb\'c7\'d2 \'bc\'f6 \'c0\'d6\'b0\'da\'b1\'ba!\
'''\
}