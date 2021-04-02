#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import torchvision
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from pylab import *

import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import glob
import os

image_size = (224,224)


# In[2]:


class kangaroo(Dataset):
    
    def __init__(self, path, transform):
        image_dir = path + '/Object_Detection_PyTorch/kangaroo/images/' 
        annotation_dir = path + '/Object_Detection_PyTorch/kangaroo/pascal/'
        self.transform = transform 
        self.image_names = os.listdir(image_dir)
        self.image_names.sort()
        self.image_names = [os.path.join(image_dir,image_name) for image_name in self.image_names]
        self.annotation_names = os.listdir(annotation_dir)
        self.annotation_names.sort()
        self.annotation_names = [os.path.join(annotation_dir,annotation_name) for annotation_name in self.annotation_names]
     
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        img = Image.open(image_name)
        w,h = img.size
        img = img.resize(image_size)
        annotation_name = self.annotation_names[idx]
        annotation_tree = ET.parse(annotation_name, ET.XMLParser(encoding='utf-8'))
        classes = annotation_tree.find("object").find("name").text 
        num_objs = 1
        boxes = []
        bndbox_xml = annotation_tree.find("object").find("bndbox")
        xmin = int(bndbox_xml.find('xmin').text)
        ymin = int(bndbox_xml.find('ymin').text)
        xmax = int(bndbox_xml.find('xmax').text)
        ymax = int(bndbox_xml.find('ymax').text)
        boxes.append([xmin*224/w,ymin*224/h,xmax*224/w,ymax*224/h])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        if self.transform is not None:
            img, target = self.transform(img, target)
        img /=255
        return img, target
    
    def __len__(self):
        return len(self.image_names)

def collate_fn(batch):
    return tuple(zip(*batch))


# In[3]:


import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


# In[ ]:


path = '/content' 
dataset = kangaroo(path, transform = get_transform(True))
print(len(dataset))


# In[4]:


train_set = int(0.8 * len(dataset)) 
val_set = len(dataset) - train_set 
print(train_set)
print(val_set)
batch_size = 3
train_data, val_data = torch.utils.data.random_split(dataset, [train_set, val_set])
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
validationloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# In[5]:


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# In[6]:


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# In[7]:


model.train()
num_epochs = 5
count = 0
for i in range(num_epochs):
  for img,targets in trainloader:
    img = list(image for image in img)
    target = [{k:v for k,v in t.items()} for t in targets]
    loss_dict = model(img,targets)
    loss = sum(loss for loss in loss_dict.values())
    loss_val = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    count = count + 1
    if (count %10 == 0):
      print("Loss at iter "+str(count)+":",loss_val)
  print("Epoch "+str(i+1)+" Loss:",loss_val)


# In[8]:


torch.save(model.state_dict(), 'kangaroo_fasterrcnn_resnet50.pth')


# In[9]:


model.eval()
img,targets = next(iter(validationloader))
img = list(image for image in img)
targets = [{k:v for k,v in t.items()} for t in targets]
output = model(img)
print(output)


# In[10]:


from matplotlib import pyplot as plt
fig, axs = plt.subplots(1, 1, figsize=(32, 16))
#axs = axs.ravel()
threshold = 0.5
sample = img[0].permute(1,2,0).numpy()
boxes = output[1]['boxes']
scores = output[1]['scores']
boxes = boxes[scores > threshold].astype(np.int32)
res_img = ImageDraw.Draw(sample)
for box in boxes:
  res_img.rectangle([box[0], box[1], box[2], box[3]],outline = "Red",width = 1)
axs[i].set_axis_off()
axs[i].imshow(sample)

