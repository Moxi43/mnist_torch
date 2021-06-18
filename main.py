import torch
from train import train
from test import test
from data_loader import getData

trainLoader = torch.utils.data.DataLoader(getData(True), batch_size=64, shuffle=True)
testLoader = torch.utils.data.DataLoader(getData(False), batch_size=1000, shuffle=True)
#for saving model as a file
modelname = "mnistnet.pth"

#training
train(trainLoader, epochs=2, model_name = modelname)

#testing and getting acc
test(modelname, testLoader)
