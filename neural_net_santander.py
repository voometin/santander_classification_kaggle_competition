import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#preprocessing
columns = train.columns[2:]
scaler = StandardScaler()
Xscaled = pd.DataFrame(scaler.fit_transform(train[columns], y=train['target']), columns=columns)

train_X, test_X, train_y, test_y = train_test_split(Xscaled, train['target'], test_size=.1, stratify=train['target'], random_state=241, shuffle=True)

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
    
class OneCycle():
    """
    In paper (https://arxiv.org/pdf/1803.09820.pdf), author suggests to do one cycle during 
    whole run with 2 steps of equal length. During first step, increase the learning rate 
    from lower learning rate to higher learning rate. And in second step, decrease it from 
    higher to lower learning rate. This is Cyclic learning rate policy. Author suggests one 
    addition to this. - During last few hundred/thousand iterations of cycle reduce the 
    learning rate to 1/100th or 1/1000th of the lower learning rate.
    Also, Author suggests that reducing momentum when learning rate is increasing. So, we make 
    one cycle of momentum also with learning rate - Decrease momentum when learning rate is 
    increasing and increase momentum when learning rate is decreasing.
    Args:
        nb              Total number of iterations including all epochs
        max_lr          The optimum learning rate. This learning rate will be used as highest 
                        learning rate. The learning rate will fluctuate between max_lr to
                        max_lr/div and then (max_lr/div)/div.
        momentum_vals   The maximum and minimum momentum values between which momentum will
                        fluctuate during cycle.
                        Default values are (0.95, 0.85)
        prcnt           The percentage of cycle length for which we annihilate learning rate
                        way below the lower learnig rate.
                        The default value is 10
        div             The division factor used to get lower boundary of learning rate. This
                        will be used with max_lr value to decide lower learning rate boundary.
                        This value is also used to decide how much we annihilate the learning 
                        rate below lower learning rate.
                        The default value is 10.
    """
    def __init__(self, num_iter, max_lr, momentum_vals=(0.95, 0.85), prcnt= 10 , div=10):
        self.num_iter = num_iter
        self.div = div
        self.step_len =  int(self.num_iter * (1- prcnt/100)/2)
        self.high_lr = max_lr
        self.low_lr = max_lr/div
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = prcnt
        self.iteration = 0
        self.lrs = []
        self.moms = []
        
    def step(self):
        self.iteration += 1
        return self.calc_lr_mom()
    
    def calc_lr_mom(self):
        if self.iteration <= self.step_len:
            ratio = self.iteration/self.step_len
            lr = self.low_lr + ratio * (self.high_lr - self.low_lr)
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        elif self.iteration <= 2 * self.step_len:
            ratio = (self.iteration - self.step_len)/self.step_len
            lr = self.high_lr - ratio * (self.high_lr - self.low_lr)
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        elif self.iteration <= self.num_iter:
            ratio = (self.iteration - 2 * self.step_len) / (self.num_iter - 2 * self.step_len)
            lr = self.low_lr - ratio * (self.high_lr - self.low_lr) / self.div
            mom = self.high_mom
        else:
            #this should never be happpened, because we have to end training here
            lr = self.high_lr/self.div
            mom = self.high_mom
        self.lrs.append(lr)
        self.moms.append(mom)
        return lr, mom

# Hyper-parameters 
architecture = [200, 600, 500, 400, 300, 1]
# architecture = [200, 200, 1]
num_epochs = 50
batch_size = 1024
optimizer_params = {
    'lr': 0.01
}

# Data loader
training_set = Dataset(train_X, train_y)
train_loader = torch.utils.data.DataLoader(dataset=training_set, 
                                           batch_size=batch_size, 
                                           shuffle=True)

validation_X_tensor = torch.Tensor(test_X.values)
train_X_tensor = torch.Tensor(train_X.values)
# train_All_tensor = torch.Tensor(Xscaled.values)
# test_X_tensor = torch.Tensor(test_scaled.values)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, architecture, device='cuda' if torch.cuda.is_available() else 'cpu', task='binary', dropoutProb=0):
        super(NeuralNet, self).__init__()
        
        self.architecture = []
        for index, layer in enumerate(architecture[:-1]):
            self.architecture.append(nn.Linear(architecture[index], architecture[index+1]))
            torch.nn.init.xavier_uniform_(self.architecture[-1].weight)
            self.architecture.append(nn.ReLU())
            if dropoutProb:
                self.architecture.append(nn.Dropout(dropoutProb))
            
        if dropoutProb:
            self.architecture.pop()
        self.architecture[-1] = nn.Sigmoid()
        self.architecture = nn.Sequential(*self.architecture)
        
        self.device = device
        self = self.to(self.device)
    
    def forward(self, out):
        out = self.architecture(out)
        return out

def prdct(model, X, mode='Test'):
    model.train(False)
    model.eval()
    with torch.no_grad():
        if model.device=='cuda':
            train_pred = model(train_X_tensor.to(model.device)).cpu().numpy().ravel()
        else:
            train_pred = model(train_X_tensor.to(model.device)).numpy().ravel()
        print('train roc_auc', roc_auc_score(train_y, train_pred), 'accuracy', accuracy_score(train_y, train_pred.round(0)))
        
        test_pred = model(X.to(model.device)).cpu().numpy().ravel()
        print('validation roc_auc', roc_auc_score(test_y, test_pred), 'accuracy', accuracy_score(test_y, test_pred.round(0)))
    
    if mode=='Test':
        return test_pred
    return roc_auc_score(test_y, test_pred)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def optimizer_params_update(optimizer, **args):
    for param_group in optimizer.param_groups:
        if args.get('lr', None):
            param_group['lr'] = args['lr']
        if args.get('momentum', None):
            param_group['momentum'] = args['momentum']

def trn(model, train_loader, num_epochs, criterion, optimizer, scheduler='', min_lr=0.00001):
    # train function
    for epoch in range(num_epochs):
        model.train(True)
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(model.device)
            batch_y = batch_y.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
        
            optimizer.step()
            
            if isinstance(scheduler, OneCycle):
                lr, mom = scheduler.step()
                optimizer_params_update(optimizer, **{'lr': lr, 'momentum': mom})
                if scheduler.iteration>=scheduler.num_iter:
                    break

        print('epoch', epoch, 'current lr', get_lr(optimizer))

        val_loss = prdct(model, validation_X_tensor, 'train')
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif isinstance(scheduler, OneCycle):
            if scheduler.iteration>=scheduler.num_iter:
                break
            
        if get_lr(optimizer)<=min_lr and not isinstance(scheduler, OneCycle):
            break
    
    print('train has been ended')

nn_model = NeuralNet(architecture, dropoutProb=0.3)
print(nn_model)
criterion = nn.BCELoss()

num_iter, max_lr = int(train_X.shape[0]/batch_size * 2), 1
scheduler = OneCycle(num_iter, max_lr)
optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-3)

# optimizer = torch.optim.Adam(nn_model.parameters(), weight_decay=1e-3, **optimizer_params)
# min_lr = 0.00001
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.2, patience=2)

trn(nn_model, train_loader, num_epochs, criterion, optimizer, scheduler)
# predictions['net'] = prdct(nn_model, validation_X_tensor)
# train_predictions['net'] = prdct(nn_model, train_All_tensor)
# test_predictions['net'] = prdct(nn_model, test_X_tensor)

# lr_finder = LRFinder(nn_model, optimizer, criterion, device=device)
# lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
# lr_finder.plot()
