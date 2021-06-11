from torchvision.datasets import MNIST


def getData(x: bool):
    return MNIST(root='./data', train=x, download = True, transform = None)
   
