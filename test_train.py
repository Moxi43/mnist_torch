from data_loader import getData
from net import Net
import torch

trainLoader = torch.utils.data.DataLoader(getData(True), batch_size=64, shuffle=True) #true - train, else - test
testLoader = torch.utils.data.DataLoader(getData(False), batch_size=1000, shuffle=True) 
net = Net() #all nn layers

def train(epochs, loss_func, optimizer):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            loss = loss_func(uputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
train(2, loss, optimizer)
