import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt 


df = pd.read_csv("dataset/largeValidation_Q3.csv")


input_size = 128
batch_size = 1024
num_classes = 10


df_train = pd.read_csv('dataset/largeTrain_Q3.csv')
df_test = pd.read_csv('dataset/largeValidation_Q3.csv')
train_y = df_train.iloc[:, 0]
train_X = df_train.iloc[:, 1:]
test_y = df_test.iloc[:, 0]
test_X = df_test.iloc[:, 1:]


class MyDataset(Dataset):
    def __init__(self, images, labels=None):
        self.X = images
        self.y = labels
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        data = np.asarray(data)
        return (data, self.y[i])


train_data = MyDataset(train_X, train_y)
test_data = MyDataset(test_X, test_y)


trainloader = DataLoader(train_data, batch_size=1024, shuffle=True)
testloader = DataLoader(test_data, batch_size=512, shuffle=True)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
#         self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
#         print(out.shape) 
#         out = self.softmax(out)
        return out



def train(model, train_iterator, test_iterator, optimizer, criterion, device, epochs = 100):
    avg_epoch = 0
    avg_test_epoch = 0
    
    for e in tqdm(range(epochs), desc = "Progress : ", position = 0, leave = True) : 
        epoch_loss = 0    
        model.train()
        epoch_test_loss = 0
        for (x, y) in train_iterator:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_pred= model(x.float())

            loss = criterion(y_pred, y)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            
            test_loss = 0
            
            for (u,v) in test_iterator:
                y_test_pred = model(u.float())
                test_loss += criterion(y_test_pred, v).item()
            test_loss = test_loss/len(test_iterator)  
            epoch_test_loss += test_loss
#         print(epoch_test_loss/len(train_iterator) ,epoch_loss / len(train_iterator) )        
        avg_test_epoch += epoch_test_loss/len(train_iterator)        
        avg_epoch +=(epoch_loss / len(train_iterator))
        
    
    
        
        
    return avg_epoch/epochs, avg_test_epoch/epochs



units = [5,20,50,100,200]
losses = []
for n in units :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, n, num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    criterion = nn.CrossEntropyLoss().to(device)
    print("Model for n = ", n)
    train_loss, test_loss = train(model,trainloader,testloader, optimizer , criterion, device)
    losses.append([train_loss, test_loss])
    print("Average Training Loss : ", train_loss)
    print("Average Validation Loss : ", test_loss)


losses = np.array(losses)
losses[:,0]

plt.plot(units,losses[:,0], label = "Average Training  Loss " )
plt.plot(units, losses[:,1], label = "Average Validation  Loss")
plt.xlabel('hidden units')
plt.ylabel('Loss')
plt.legend()
plt.show()


lrs = [0.1,0.01,0.001]
history = {}
for l in lrs :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, 4, num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr = l)
    criterion = nn.CrossEntropyLoss().to(device)
    print("Model for l = ", l)
    train_loss, test_loss = train(model,trainloader,testloader, optimizer , criterion, device)
    history[l] = (train_loss, test_loss)
    print("Average Training Loss : ", sum(train_loss)/len(train_loss))
    print("Average Validation Loss : ", sum(test_loss)/len(test_loss))


for key in list(history.keys()):
    plt.plot([x for x in range(1,101,1)],history[key][0], label = "Training  Loss " )
    plt.plot([x for x in range(1,101,1)], history[key][1], label = "Validation  Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epochs (lr = ' + str(key) + " )")
    plt.legend()
    plt.show()