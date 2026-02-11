# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Write your own steps

### STEP 2: 



### STEP 3: 



### STEP 4: 



### STEP 5: 



### STEP 6: 





## PROGRAM
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

data=pd.read_csv(r"C:\Users\admin\Documents\DEEP LEARNING\data\iris.csv")
data
class Model(nn.Module):
    
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3): # How many Layers?

        # Input Layers (4 futures) ---> h1 N--> N---> output (3 class)
        
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)    # input layer
        self.fc2 = nn.Linear(h1, h2)            # hidden layer
        self.out = nn.Linear(h2, out_features)  # output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
# Instantiate the Model class using parameter defaults:
torch.manual_seed(32)
model = Model()
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv(r"C:\Users\admin\Documents\DEEP LEARNING\data\iris.csv")
df.head()
df.tail()
X = df.drop('target',axis=1)
y = df['target']
X = X.values
y = y.values
type(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=33)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
criterion = nn.CrossEntropyLoss()  # multi class clasification problem 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # 0.000001
model.parameters
epochs = 100
losses = []

for i in range(epochs):
    
    i+=1

    # Forward and get a prediction 
    
    y_pred = model.forward(X_train)

    #Calculation Loss/error
    loss = criterion(y_pred, y_train)
    
    losses.append(loss.item())
    
    # a neat trick to save screen space:
    if i%10 == 1:
        print(f'epoch: {i}  loss: {loss.item()}')

    #Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs), losses)

plt.ylabel('Loss')
plt.xlabel('epoch');
correct = 0

with torch.no_grad():

    for i,data in enumerate(X_test):

        y_val =model.forward(data)

        #1.) 2.) 3.)

        print(f'{i+1}.) {str(y_val)} {y_test[i]}')

        # 0 1 2

        if y_val.argmax().item() == y_test[i]:
            correct +=1

print(f'We got {correct} correct!')
print(f'{i+1}.) {str(y_val.argmax().item())} {y_test[i]}')
torch.save(model.state_dict(),'my_iris_model.pt')



```

### Name:DHINESH S

### Register Number:212224220025

```python
class IrisClassifier(nn.Module):
    def __init__(self, input_size):
        super(IrisClassifier, self).__init__()
        #Include your code here

    def forward(self, x):
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    #Include your code here

```

### Dataset Information
Include screenshot of the dataset.

### OUTPUT

## Confusion Matrix

Include confusion matrix here
### RESULT
Thus, a neural network classification model was successfully developed and trained using PyTorch


