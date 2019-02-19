import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, X, y):
        'Initialization'
        self.y = torch.Tensor(y.values.reshape(-1, 1))#, dtype=torch.long)
        self.X = torch.Tensor(X.values)

    def __len__(self):
        'Denotes the total number of samples'
        return self.X.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index]

columns = train.columns[2:]
# Xscaled = StandardScaler().fit_transform(train[columns], y=train['target'])
Xscaled = pd.DataFrame(MinMaxScaler().fit_transform(train[columns], y=train['target']), columns=columns)

train_X, test_X, train_y, test_y = train_test_split(Xscaled, train['target'], test_size=.1, stratify=train['target'], random_state=241, shuffle=True)
# train_X.shape, train_y.shape, test_X.shape, test_y.shape

# Hyper-parameters 
architecture = [200, 200, 200, 1]
num_epochs = 10
batch_size = 256
optimizer_params = {
    'lr': 0.05
}

# Data loader
training_set = Dataset(train_X, train_y)
train_loader = torch.utils.data.DataLoader(dataset=training_set, 
                                           batch_size=batch_size, 
                                           shuffle=True)

validation_X = torch.Tensor(test_X.values)
train_X_tensor = torch.Tensor(train_X.values)
# validation_X = torch.Tensor(test_X.values)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, architecture, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(NeuralNet, self).__init__()
        
        self.architecture = []
        for index, layer in enumerate(architecture[:-1]):
            self.architecture.append(nn.Linear(architecture[index], architecture[index+1]))
            torch.nn.init.xavier_uniform_(self.architecture[-1].weight)
            self.architecture.append(nn.ReLU())
            
        self.architecture[-1] = nn.Sigmoid()
        self.architecture = nn.Sequential(*self.architecture)
        
        self.device = device
        self = self.to(self.device)
    
    def forward(self, out):
        out = self.architecture(out)
        return out

def prdct(model, X):
    model.train(False)
    model.eval()
    with torch.no_grad():
        train_pred = model(train_X_tensor.to(model.device)).cpu().numpy().ravel()
        print('train roc_auc', roc_auc_score(train_y, train_pred), 'accuracy', accuracy_score(train_y, train_pred.round(0)))
        test_pred = model(X.to(model.device)).cpu().numpy().ravel()
        print('validation roc_auc', roc_auc_score(test_y, test_pred), 'accuracy', accuracy_score(test_y, test_pred.round(0)))

def trn(model, train_loader, num_epochs, criterion, optimizer):
    print(model.architecture[-2].weight)
    for epoch in range(num_epochs):
        model.train(True)
        flag = False
        for batch_X, batch_y in train_loader:
            if not flag:
                flag = True
                print('new_epoch')
            batch_X = batch_X.to(model.device)
            batch_y = batch_y.to(model.device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
        
            optimizer.step()
        print(model.architecture[-2].weight)

        if epoch%10==0:
            print(epoch)

        prdct(model, validation_X)
    print('train has been ended')


model = NeuralNet(architecture)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
trn(model, train_loader, num_epochs, criterion, optimizer)
# lr_finder = LRFinder(model, optimizer, criterion, device=device)
# lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
# lr_finder.plot()

