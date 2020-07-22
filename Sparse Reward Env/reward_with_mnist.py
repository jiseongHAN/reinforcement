import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

train_dataset = datasets.MNIST('.../mnist_data/', download=True)

'''
dir(train_dataset)
'''
train_data = train_dataset.train_data
train_label = train_dataset.train_labels

'''
train_data
train_label
'''

'''
make target -> predict net
'''

class RNetwork(nn.Module):
    def __init__(self, n_input):
        super(RNetwork, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        self.target = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.target.apply(self.random_init)

    def random_init(self, m): # for target_network
        if type(m) == nn.Linear:
            nn.init.orthogonal_(m.weight)

    def forward(self,x):
        pred = self.pred(x)
        target = self.target(x)
        return pred, target



def train(pred, target, opt):
    mse = nn.MSELoss()
    loss = mse(target.detach(),pred)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return torch.mean(loss) / pred.shape[0]


'''
test with some data from mnist
'''
epoch = 10

net = RNetwork(28*28)

test_data = train_data[:100,].reshape(100,-1).float()
new_data = train_data[100:105,].reshape(5,-1).float()
opt = optim.Adam(net.pred.parameters(),lr=0.0005)
pred, target = net(new_data)

for i in range(epoch):
    loss = train(pred, target, opt)
    print('epi : %d Loss: %f' %(i,loss))
    pred, target = net(test_data)

pred, target = net(new_data)
loss = train(pred, target, opt)
print('epi : Extra Loss: %f' %(loss))
