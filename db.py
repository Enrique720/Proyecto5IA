import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np 
import math 
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
#   

# [1,1,1    ,1,     1,1]
#  emotions,gender,age
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

emotions_dictonary = {
    'a': 0, 'd': 1, 'f':2 , 'h': 3, 'n': 4, 's': 5 
}
age_dictionary = {
    'y': 0 , 'm': 1, 'o': 2    
}



def oneRealHotEncoder(emotion,sex,age):
    string = "{0:b}".format(emotions_dictonary[emotion]).zfill(3)
    if sex == 'm':
        string +='1'
    else: 
        string +='0'     
    string +="{0:b}".format(age_dictionary[age]).zfill(2)
    return [int(i) for i in string]

def oneNotRealHotEncoder(emotion,sex,age):
    emotions = ['a','d','f','h','n','s']
    sexs = ['m','f']
    ages = ['y','m','o']
    #return_value = np.zeros(len(emotions)*len(sexs)*len(ages))
    #return_value = np.zeros(len(emotions))
    emotion_index = emotions.index(emotion)
    #print(emotion_index)
    sex_index = sexs.index(sex)
    #print(sex_index)
    age_index = ages.index(age)
    #print(age_index)
    #return_index =emotion_index*len(sexs)*len(ages)+sex_index*len(ages)+age_index 
    #return_value[emotion_index*len(sexs)*len(ages)+sex_index*len(ages)+age_index]=1
    return_index =emotion_index

    return  return_index




def get_imgs():
    dir = './db/'
    images = glob.glob(dir+'*')
    return images 

def process_image(filename):
    sub_data = filename.split("_")
    age = sub_data[1]
    sex = sub_data[2]
    emotion = sub_data[3]
    labels =oneRealHotEncoder(emotion,sex,age)
    img = Image.open(filename)
    w,h = img.size
    img = np.array(img.getdata())/255
    img2 = [img[:,i].reshape(w,h) for i in range(len(img[0]))]
    img2 = np.array(img2)
    img2 = torch.tensor(img2).float()
    labels = torch.tensor(labels).float() 
    #img2 = torch.tensor(img2).float()
    return (img2,labels)

def process_image2(filename):
    sub_data = filename.split("_")
    age = sub_data[1]
    sex = sub_data[2]
    emotion = sub_data[3]
    labels =oneNotRealHotEncoder(emotion,sex,age)
    img = Image.open(filename)
    img = img.resize((224,280),Image.ANTIALIAS)
    # Imagen: 2835 x 3543 todas 
    #224 
    w,h = img.size
    img = np.array(img.getdata())
    img2 = [img[:,i].reshape(w,h) for i in range(3)]
    img2 = np.array(img2)
    #print(img2)
    img2 = torch.tensor(img2).float()
    std,mean = torch.std_mean(img2)
    transform_norm = transforms.Compose([
        transforms.Normalize(mean, std),
    ])
    img2 = transform_norm(img2)
    #print(img2)
    #print(img2.shape)
    labels = torch.tensor(labels)
    #img2 = torch.tensor(img2).float()
    return (img2,labels)

#imgs = get_imgs()
#print(process_image2(imgs[2]))
