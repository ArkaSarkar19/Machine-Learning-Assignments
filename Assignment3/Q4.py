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
import random
import pickle
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


f = open("dataset/CIFAR-10/train_CIFAR.pickle", "rb")
train = pickle.load(f)
f.close()
f = open("dataset/CIFAR-10/test_CIFAR.pickle", "rb")
test = pickle.load(f)
f.close()


train_X = train["X"]
train_y = train["Y"]
test_X = train["X"]
test_y = train["Y"]


def plot_images(n_images, images, labels):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))
    fig = plt.figure(figsize = (10, 10))
    
    for i in range(rows*cols):
        img = train_X[i].reshape(3,32,32)
        img = img.transpose(1, 2, 0)
        
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(img)
        ax.set_title(labels[i])
        ax.axis('off')
        

plot_images(20, train_X, train_y)


from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(train_X)

X = pca.transform(train_X)

from matplotlib import pyplot as plt
plt.figure( figsize=(15,15) )
plt.scatter(X[:, 0], X[:, 1], c=train_y)
plt.show()


new_train_X = []


for i in tqdm(range(len(train_X)), leave = True, position = 0):
    im = train_X[i,:]
    img = im.reshape(32, 32, 3)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, np.float32)
    new_train_X.append(img)


new_test_X = []

for i in tqdm(range(len(test_X)), leave = True, position = 0):
    im = test_X[i,:]
    img = im.reshape(32, 32, 3)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, np.float32)
    new_test_X.append(img)


def calculate_class_distribution(labels):
    n = len(labels)
    classes = np.unique(labels)
    for i in classes:
        curr = np.count_nonzero(labels == i)
        print("Class frequency of ", i , " is ", str(curr), "/", str(n))


calculate_class_distribution(np.asarray(train_y))

alexnet = models.alexnet(pretrained=True)

model = nn.Sequential(*list(alexnet.classifier.children()))
alexnet.classifier = model     


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


alexnet.to(device)


class MyDataset(Dataset):
    def __init__(self, images, labels, transforms = None):
        self.X = images
        self.y = labels
        self.transform = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i,:]
        img = Image.fromarray(data)
        
        if self.transform is not None:
            # print("dffdfdf")
            return self.transform(img), self.y[i]


        return img, self.y[i]


train_transform_aug = transforms.Compose([
    transforms.Resize((40, 40)),
    transforms.Pad(16),       
    transforms.ToTensor(),
])

train_data = MyDataset(train_X, train_y, train_transform_aug)
test_data = MyDataset(test_X, test_y, train_transform_aug)


trainloader = DataLoader(train_data, batch_size=1, shuffle=True)
testloader = DataLoader(test_data, batch_size=1, shuffle=True)


train_extracted_x = []
train_extracted_y = []
for (x, y) in tqdm(trainloader, desc = "Progress : ", leave = True, position = 0 , total = len(trainloader)) :

    x = x.to(device)
    y = y.to(device)
    y_pred= alexnet(x.float())


    train_extracted_x.append(y_pred)
    train_extracted_y.append(y)
    del x
    del y

train_feature = {"X" : train_extracted_x, "Y" : train_extracted_y }
f = open("cifar_features_train", "wb")
pickle.dump(train_feature, f)
f.close()


test_extracted_x = []
test_extracted_y = []
for (x, y) in tqdm(testloader, desc = "Progress : ", leave = True, position = 0 , total = len(testloader)) :

    x = x.to(device)
    y = y.to(device)
    y_pred= alexnet(x.float())


    test_extracted_x.append(y_pred)
    test_extracted_y.append(y)
    del x
    del y
test_features = {"X" : test_extracted_x, "Y" : test_extracted_y }
f = open("cifar_features_test", "wb")
pickle.dump(test_features, f)
f.close()


new_train = pickle.load(open("dataset/CIFAR-10/cifar_features_train", "rb"))
new_test = pickle.load(open("dataset/CIFAR-10/cifar_features_test", "rb"))


new_train_X = np.squeeze(new_train["X"])
new_train_y = new_train["Y"].reshape(1,-1).T
new_test_X = np.squeeze(new_test["X"])
new_test_y = new_test["Y"].reshape(1,-1).T


import tensorflow as tf


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(512,input_shape = (1000,),activation='relu'),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=['accuracy'],
)

model.fit(
    new_train_X, new_train_y,
    epochs=100
    
)


y_pred = model.predict(new_test_X)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


y_pred[y_pred>=0.5] = 1
y_pred[y_pred<0.5] = 0
print(y_pred)


cm = confusion_matrix(y_pred, new_test_y)
print(cm)


acc = accuracy_score(y_pred, new_test_y)
print(acc)