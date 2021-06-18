from torchvision.datasets import MNIST
from os.path import isdir
from torchvision import transforms

def getData(x: bool):
    '''
    bool x - train or not parameter:
    if true, return train dataset, else, return test dataset.

    '''
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    if isdir("./data"):
        return MNIST(root='./data', train=x, download = False, transform = transform)
    else:
        return MNIST(root='./data', train=x, download = True, transform = transform)
