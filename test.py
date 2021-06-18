from net import Net
import torch


def test(model_name, loader): 
    net = Net()
    net.load_state_dict(torch.load("./"+model_name))
    dataiter = iter(loader)
    images, labels = dataiter.next()
    #to be able to count the acc
    correct = 0
    total = 0 

    with torch.no_grad(): #no gradients cuz' it's not training
        for data in loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy: %d %%" % (100 * correct / total))

            
                
    
    
