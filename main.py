import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transformers 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer=transformers.Compose([
    transformers.ToTensor(),
    transformers.Normalize((0.1307),(0.3081))
])
MNIST_train_Data=torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=False,
    transform=transformer
)
MNIST_test_Data=torchvision.datasets.MNIST(
    root="./data_test",
    train=False,
    download=False,
    transform=transformer
)
train_loader=DataLoader(MNIST_train_Data,shuffle=True,batch_size=64)
test_loader=DataLoader(MNIST_test_Data,shuffle=False,batch_size=1000)
class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network,self).__init__()
        self.flaten=nn.Flatten()
        self.layers=nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x=self.flaten(x)
        x=self.layers(x)
        return x
   
model=Neural_Network()
loss_function=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model,loss_function,optimizer,train_loader,device):
    model=model.to(device)
    model.train()
    epoch_loss=0.0
    running_loss=0.0
    total_batches=len(train_loader)
    for batch_idx,(inputs,targets) in enumerate(train_loader):
        num_correct_predictions=0.0
        total_predictions=0.0
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output=model(inputs)
        loss=loss_function(output,targets)
        loss.backward()
        optimizer.step()
        loss_value=loss.item()
        epoch_loss+=loss_value
        running_loss+=loss_value
        _, predicted_indices = output.max(1)
        batch_size = targets.size(0)
        total_predictions += batch_size
        num_correct_in_batch = predicted_indices.eq(targets).sum().item()
        num_correct_predictions += num_correct_in_batch 
        if (batch_idx + 1) % 134 == 0 or (batch_idx + 1) == total_batches:
            avg_running_loss = running_loss / 134
            accuracy = 100. * num_correct_predictions / total_predictions
            print(f'\tStep {batch_idx + 1}/{total_batches} - Loss: {avg_running_loss:.3f} | Acc: {accuracy:.2f}%')
            running_loss = 0.0
            num_correct_predictions = 0
            total_predictions = 0
    avg_epoch_loss = epoch_loss / total_batches
    return model, avg_epoch_loss   

def evaluate(model, test_loader, device):
    model.eval()
    num_correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted_indices = outputs.max(1)
            batch_size = targets.size(0)
            total_predictions = total_predictions + batch_size
            correct_predictions = predicted_indices.eq(targets)
            num_correct_in_batch = correct_predictions.sum().item()
            num_correct_predictions = num_correct_predictions + num_correct_in_batch
    accuracy_percentage = (num_correct_predictions / total_predictions) * 100
    print((f'\tAccuracy - {accuracy_percentage:.2f}%'))
    return accuracy_percentage

train_loss=[]
train_acc=[]

for epoch in range(6):
    print(f'\nTraining Epoch {epoch+1}:')
    trained_model,loss=train_epoch(model,loss_function,optimizer,train_loader,device)
    train_loss.append(loss)
    print(f'Testing Epoch {epoch+1}:')
    accuracy=evaluate(model, test_loader, device)
    train_acc.append(accuracy)





















































