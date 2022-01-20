from net import Net
import torch


def train(loader, epochs, model_name):
    net = Net()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            #params
            loss_func = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
            ######
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print("epoch: " + str(epoch), "i: " + str(i), "loss: " + str(running_loss/(i+1)))
            if i%2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0
    #saving the model
    torch.save(net.state_dict(), "./" + model_name)

