#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.models as models
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets


# In[2]:


resnet50 = models.resnet50(pretrained=True)


# In[3]:


from EarlyExitModel import EarlyExitModel

model = EarlyExitModel(resnet50, 1000)
model


# In[4]:


model.clear_exits()
exit_layers = [model.add_exit(layer) for layer in ('layer1', 'layer2', 'layer3')]
model


# In[5]:


# dataset = datasets.ImageFolder('./imagenet1k')


# In[8]:


hf_dataset = load_dataset("frgfm/imagenette", '320px')
hf_dataset = concatenate_datasets(hf_dataset.values())

class CustomDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        record = self.hf_dataset[idx]
        sample = record['image']
        if self.transform is not None:
            sample = self.transform(sample)
        label = record['label']
        print(sample, label)
        return sample, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


torch_dataset = CustomDataset(hf_dataset, transform=transform)

batch_size = 32

test_size = 0.2
test_volume = int(test_size * len(torch_dataset))
train_volume = len(torch_dataset) - test_volume

train_dataset, test_dataset = random_split(torch_dataset, [train_volume, test_volume])
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    # num_workers=4
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    # num_workers=4
)


# In[10]:


for layer in exit_layers:
    layer.force_forward()

for X, y in train_dataloader:
    y_hat, exits = model(X)
    print('exits',exits)
    print(y_hat)
    break


# In[ ]:





# In[ ]:





# In[12]:


# get_ipython().system('{sys.executable} -m jupyter nbconvert --to script main.ipynb')


# In[ ]:





# In[ ]:




