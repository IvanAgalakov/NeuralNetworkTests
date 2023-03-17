import torch
import torch.nn as nn
import random


def bitfield(n):
    return [int(digit) for digit in bin(n)[2:]] # [2:] to chop off the "0b" part 

class MultiLayerNet(torch.nn.Module):
    def __init__(self):
        super(MultiLayerNet, self).__init__()
        self.input_layer=nn.Linear(32,16,dtype=torch.float64)
        self.hidden_layer1=nn.Linear(16,8,dtype=torch.float64)
        self.hidden_layer2=nn.Linear(8,4,dtype=torch.float64)
        # self.hidden_layer3=nn.Linear(64,32,dtype=torch.float64)
        self.output_layer=nn.Linear(4,2,dtype=torch.float64)

    def forward(self,input):
        h1=torch.relu(self.input_layer(input))
        h2=torch.sigmoid(self.hidden_layer1(h1))
        h3=torch.tanh(self.hidden_layer2(h2))
        # h4=torch.tanh(self.hidden_layer3(h3))
        output=self.output_layer(h3)
        return output


model=MultiLayerNet()

loss_function=nn.CrossEntropyLoss()

train_data = []
train_labels = []

n = 10000
for i in range(0,n):
    ran = random.randint(int(pow(2,31)/n)*i,int(pow(2,31)/n)*(i+1))
    while(ran in train_data):
        ran = random.randint(1,pow(2,31))
    first = ran
    bits = bitfield(first)
    bits.extend([0]*(32-len(bits)))
    train_data.append(bits)
    if (first)%2 == 0:
        train_labels.append([0,1])
    else:
        train_labels.append([1,0])

test_data = []
test_labels = []
for i in range(0,100):
    first = random.randint(int(pow(2,31)/100)*i,int(pow(2,31)/100)*(i+1))
    bits = bitfield(first)
    bits.extend([0]*(32-len(bits)))
    test_data.append(bits)
    if (first)%2 == 0:
        test_labels.append([0,1])
    else:
        test_labels.append([1,0])

train_data=torch.tensor(train_data,dtype=torch.float64)
test_data=torch.tensor(test_data,dtype=torch.float64)

train_labels=torch.tensor(train_labels,dtype=torch.float64)

from torch.optim import Adam

optimizer=Adam(model.parameters(),lr=0.0001)

from tqdm import tqdm
import numpy as np

n_epoch=5

for epoch in range(n_epoch):
    all_loss=list()
    print("on Epoch ", epoch+1)
    print("Epoch#",epoch+1,end=' ')
    for i in tqdm(range(len(train_data))):
        model.zero_grad()
        
        pred=model(train_data[i])
        #print(pred)
        loss=loss_function(pred,train_labels[i])
        
        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())
        
    print("\nAverage loss=",np.mean(all_loss))




with torch.no_grad():
    all_pred=list()
    all_target=list()
    for i in range(len(test_data)):
        pred=model(test_data[i])
        pred=torch.argmax(pred) # determinig the output with largest value
        all_pred.append(pred.item())
        all_target.append(test_labels[i])


correct = 0    
for i in range(0,len(all_pred)):
    if(all_target[i][all_pred[i]] == 1):
        correct += 1

print(str((correct/len(all_pred))*100), "% correct")

while(True):
      x = input("enter a number to the model: ")
      if(x == "exit"):
          break
      x = int(x)
      bits = bitfield(x)
      bits.extend([0]*(32-len(bits)))
      run=torch.tensor(bits,dtype=torch.float64)
      result = model(run)
      pred=torch.argmax(result) # determinig the output with largest value
      print("result is - ", pred)
