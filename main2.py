import torch
from torch import optim
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math
from db2 import get_imgs,process_image,process_image2

# Imagen: 2835 x 3543 todas 
#224 
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential(
                                nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=4,stride=2)
    )
    self.conv12 = nn.Sequential(
                                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=1),
                                nn.ReLU()
    )
    self.conv2 = nn.Sequential(
                                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=4,stride=2)
    )
    self.conv3 = nn.Sequential(
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=4,stride=2)
    )
    self.conv4 = nn.Sequential(
                                 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=4,stride=2)
    )
    self.conv5 = nn.Sequential(
                                 nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=4,stride=2)
    )
    self.fc = nn.Sequential(
                              nn.Linear(in_features=256*4*6, out_features=1536),
                              nn.ReLU(),
                              nn.Linear(in_features=1536, out_features=384),
                              nn.ReLU(),
                              nn.Linear(in_features=384, out_features=96),
                              nn.ReLU(),
                              nn.Linear(in_features=96, out_features=36),
                              nn.Softmax(),
    )
    


    # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=4, stride=3, padding=1)
    # self.conv3 = nn.Conv2d(in_channels=64*2, out_channels=64*2*2, kernel_size=4, stride=2, padding=2)
    # self.fc = nn.Linear(in_features=64*2*2*158*198, out_features=6)

    # self.layer1 = nn.Sequential(
    #         nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2))
    #     self.layer2 = nn.Sequential(
    #         nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
    #         nn.ReLU(),
    #         nn.MaxPool2d(kernel_size=2, stride=2))
    #     self.fc = nn.Linear(7*7*32, num_classes)
        

  def forward(self, image):
    out = self.conv1(image)
    out = self.conv12(out)
    #print(out.shape)
    
    out = self.conv2(out)
    #print(out.shape)

    out = self.conv3(out)
    #print(out.shape)

    out = self.conv4(out)
    #print(out.shape)

    out = self.conv5(out)
    

    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    #print("3", out.shape)
    
    # out = out.view(out.size(0), -1)
    return out


batch_size = 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(device)

imgs =get_imgs()

train_loader = torch.utils.data.DataLoader(dataset=imgs, batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(dataset=imgs, batch_size=batch_size, shuffle=False)

def test(model,filenames):
  with torch.no_grad(): 
    goods = 0
    bads = 0
    for filename in filenames:
      image,labels = process_image2(filename)
      image = image.unsqueeze(0) #[3][2000][3000]
      image = image.to(device)
      labels = labels.unsqueeze(0)
      labels = labels.to(device)
      output = model(image)
      output_label = torch.argmax(output)
      if(output_label == labels):
        goods +=1
      else:
        bads +=1
    print("goods are: ",goods)
    print("bads are: ",bads)
  
def train(model, optimizer, loss_fn, num_epochs):
  
  loss_vals = []
  running_loss =0.0
  # train the model

  list_loss= []
  list_time = []
  j=0

  for epoch in range(num_epochs):
    filenames = next(iter(train_loader))
    i =0          
    for i,filenames in enumerate(train_loader):
      for filename in filenames:
        image,labels = process_image2(filename)
        image = image.unsqueeze(0) #[3][2000][3000]
        image =image.to(device)
        labels = labels.unsqueeze(0)
        labels = labels.to(device)
        #print(labels.shape)
        # forward 
        output = model(image)
        #print(output)
        #print(labels )
        print(output)
        print(labels)
        loss   = loss_fn(output, labels)
        # change the params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
        list_loss.append(loss.item())
        list_time.append(j)
        j+=1
      if (i+1) % 1 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs,loss.item()))
              
  print('Finished Training Trainset')
  return list_loss
  
learning_rate = 0.0007
epochs = 40
cnn = CNN().to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=learning_rate)

#print(data.shape)
#print(cnn(data))
train(cnn,optimizer,loss,epochs)
test(cnn,imgs)

