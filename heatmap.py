# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:13:14 2023

@author: AmayaGS
"""

# %%

import os
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
import openslide as osi

from patchify import patchify, unpatchify

from sklearn.model_selection import train_test_split

Image.MAX_IMAGE_PIXELS = None

# %%

patch_size = 64
step = 64
slide_level = 1
main_path = r"placeholder for slide path"

# %%

slide = osi.OpenSlide(main_path)
properties = slide.properties
#adjusted_level = int(slide_level + np.log2(int(properties['openslide.objective-power'])/40))
slide_adjusted_level_dims = slide.level_dimensions[slide_level]
np_img = np.array(slide.read_region((0, 0), slide_level, slide_adjusted_level_dims).convert('RGB'))

# %%

np_mask = cv2.imread(np_img, 0)
np_mask_resized = cv2.resize(np_img, slide_adjusted_level_dims)

# %%

patches_img = patchify(np_img, (patch_size, patch_size, 3), step=step)
patches_mask = patchify(np_mask_resized, (patch_size, patch_size), step=step)

# %%

from attention_models import VGG_embedding, GatedAttention

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_net = VGG_embedding()
gated_net = GatedAttention()
        
# %%

patient_embedding = []
image = []

for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            image.append(patches_img[i, j, 0, :, :, :])
            
# %%

label = torch.Tensor(1)
patient_embedding = []

for im in image:
    
    # #if single_patch_mask.any() == 255:
    tensor = test_transform(Image.fromarray(im))
    embedding = embedding_net(tensor.unsqueeze(0).cuda())
    embedding = embedding.detach().to('cpu')
    embedding = embedding.squeeze(0)
    patient_embedding.append(embedding)
    
patient_embedding = torch.stack(patient_embedding)
    
logits, Y_prob, Y_hat, A, instance_dict = gated_net(patient_embedding.cuda(), label=label, instance_eval=False)

# %%

weights = A.detach().to('cpu').numpy()

# %%

attention_list = []

for i in range(len(weights[0])):
    att_arr = np.full(shape=(64, 64), fill_value=weights[0][i])
    attention_list.append(att_arr)
    
attention_array  = np.concatenate(attention_list, axis=1)
att_img = np.zeros(np_img.shape[:2])
    
# %%

step1 = 64
step2 = 5376 # image size
# range is equal to im_size / 224

for i in range(84):
    att_img[i*step1: i*step1 + step1, ] += attention_array[0:64, i*step2 : i*step2 + step2]
    
# %%

att_std = (att_img - att_img.mean()) / (att_img.std())
plt.matshow(att_std*255, cmap=plt.cm.RdBu)
plt.axis('off')
plt.show()

# %%

plt.figure(figsize=(50, 50))
plt.subplot(221)
plt.title('Original image', size=50)
plt.imshow(np_img)
plt.subplot(222)
plt.title('Predicted heatmap', size=50)
plt.imshow(att_std, cmap=plt.cm.RdBu)
plt.show()

# %%

plt.figure(figsize=(50, 50))
plt.imshow(att_std, cmap=plt.cm.RdBu)
plt.title('Heatmap overlay', size=50)
plt.imshow(np_img, alpha=0.8)
plt.show()

# %%
