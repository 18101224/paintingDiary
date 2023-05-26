import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from PIL import Image
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import glob
from torch import mps
import cv2 as cv
from sklearn.cluster import KMeans
import random
import tkinter as tk
from tkinter import filedialog
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

#setting
size = (512,512)
path = './processed/'
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
path_list = sorted(glob.glob('./processed/*.jpg'))# for the style mean
random.shuffle(path_list)
def get_image(path, img_transform, size = (512,512)):
    image = Image.open(path)
    image = image.resize(size, Image.LANCZOS)
    image = img_transform(image).unsqueeze(0)
    return image.to(device)

def get_gram(m):
    _, c, h, w = m.size()
    m = m.view(c, h * w)
    m = torch.mm(m, m.t()) 
    return m

def denormalize_img1(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = inp.transpose((2,0,1))
    return inp
def denormalize_img(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.selected_layers = [3, 8, 15, 22]
        self.vgg = models.vgg16(pretrained=True).features
        
    def forward(self, x):
        layer_features = []
        for layer_number, layer in self.vgg._modules.items():
            x = layer(x)
            if int(layer_number) in self.selected_layers:
                layer_features.append(x)
        return layer_features
def compute_mean(path_list,encoder,sample_num):
    sample = get_image(path_list[0],img_transform)
    sample = encoder(sample)
    style_features =[]
    for i,feature in enumerate(sample):
        style_features.append(torch.zeros_like(get_gram(feature)))
    n=0
    for i in range(sample_num):
        n+=1
        idx = np.random.randint(0,len(path_list))
        style_img = get_image(path_list[idx],img_transform)
        style_feature = encoder(style_img)
        for index, feature in enumerate(style_feature):
            _,c,h,w  = feature.size()
            gram = get_gram(feature)
            style_features[index] = ((style_features[index]*((n-1)/n)) + (1/n)*gram)

    return style_features
def tranfer(encoer,content_feature,style_features,isReopt=False):
    global n_epoch
    global generated_img
    global optimizer
    global style_weight
    global content_weight
    for epoch in range(n_epochs):
        generated_features = encoder(generated_img)
        content_loss = torch.mean((content_feature[-1] - generated_features[-1])**2)
        style_loss = 0
        for gf, sf in zip(generated_features, style_features):
            _, c, h, w = gf.size()
            gram_gf = get_gram(gf)
            gram_sf = get_gram(sf)
            style_loss += torch.mean((gram_gf - gram_sf)**2)  / (c * h * w)
        loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        loss.backward(
        )
def viz(output,original):
    output = output.detach().cpu().squeeze()
    output= denormalize_img(output)
    original = original.detach().cpu().squeeze()
    original = denormalize_img(original)
    merged= np.hstack((original,output))
    print(merged.shape,merged.dtype)
    merged = np.float32(merged)
    merged = cv.cvtColor(merged,cv.COLOR_BGR2RGB)
    cv.imshow('the original and result',merged)
    cv.waitKey()
def kmeans(path):
    global device
    sample =  Image.open(path)
    sample = sample.resize(size, Image.LANCZOS)
    sample = img_transform(sample) # 3 300 400 (channel, height,width)
    sample= sample.transpose(0,2)
    original_shape = sample.shape
    print(sample.shape)
    kmeans = KMeans(n_clusters=12)
    flattened_img = torch.reshape(sample,(-1,3)) # 300*400 X 3
    kmeans.fit(flattened_img)
    zipped_label = kmeans.predict(flattened_img)
    flattened_img = torch.tensor(kmeans.cluster_centers_[zipped_label],dtype=torch.float32)
    sample = torch.reshape(flattened_img,original_shape)
    sample = sample.transpose(0,2) #이게 최종 불러온 이미지
    print(sample.shape)
    sample = torch.tensor(denormalize_img1(sample))
    print(sample.shape)
    vutils.save_image(sample,'temp.jpg')
def open_file_dialog():
    # Create the Tkinter root window
    root = tk.Tk()
    # Hide the root window
    root.withdraw()

    # Show the file dialog and wait for the user's selection
    file_path = filedialog.askopenfilename()

    # Check if the user selected a file
    if file_path:
        # Print the selected file path
        print("Selected file:", file_path)
        # You can perform further actions with the file path here
    return file_path
if __name__ == '__main__':
    encoder = FeatureExtractor().to(device)
    for p in encoder.parameters():
        p.requires_grad=False
    
    style_features = torch.load('style_features.pt')

    content_weight, style_weight = 300,10000
    n_epochs = 500

    original_path = open_file_dialog()
    #now optimize the content for the style
    start_time = time.time()
    cv.namedWindow("Please wait", cv.WINDOW_AUTOSIZE)
    kmeans(original_path)
    content_img = get_image('temp.jpg',img_transform)
    content_img*=1.5
    generated_img = content_img.clone()
    generated_img.requires_grad = True
    optimizer = torch.optim.Adam([generated_img],lr=0.003,betas=[0.5, 0.999])
    content_features = encoder(content_img)
    for epoch in range(n_epochs):
        generated_features = encoder(generated_img)
        
        content_loss = torch.mean((content_features[-1] - generated_features[-1])**2)  #MSE
        style_loss = 0
        i=1
        for gf, sf in zip(generated_features, style_features):
            _, c, h, w = gf.size()
            gram_gf = get_gram(gf)
            style_loss += np.exp(i**4)*torch.mean((gram_gf - sf)**2)  / (c * h * w)

        loss = content_weight*content_loss + style_weight*style_loss
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(epoch)
            print(style_loss,content_loss)
            mps.empty_cache()
    cv.destroyAllWindows()
    content_img = get_image(original_path,img_transform)
    viz(generated_img,content_img)