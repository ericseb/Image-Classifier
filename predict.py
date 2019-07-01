import numpy as np
import pandas as pd
import seaborn as sb
import json
import time
import copy

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from PIL import Image
from collections import OrderedDict
import argparse

#SET DEFAULT VALUES
checkpoint = 'checkpoint.pth'
filepath = 'cat_to_name.json'    
arch=''
image_path = 'flowers/test/100/image_07896'
top_k = 5

#ARGSPARSE
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', action='store',type=str, help='Checkpoint of trained model.')
parser.add_argument('--image_path',action='store',type=str, help='Images location path')
parser.add_argument('--top_k', action='store',type=int, help='Number of classes you wish to see in descending order.')
parser.add_argument('--json', action='store',type=str, help='json file containing class names')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
args = parser.parse_args()

#LOAD MDOEL
def load_model(checkpoint_path):

    checkpoint = torch.load(checkpoint_path)
    
    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
        in_features = 25088
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = 9216
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = 1024
        for param in model.parameters():
            param.requires_grad = False
    else:
        print('Sorry base architecture not recognised')
    
    model.class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_units']
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, hidden_units)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(hidden_units, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

# Select parameters entered in command line
if args.checkpoint:
    checkpoint = args.checkpoint
if args.image_path:
    image_path = args.image_path
if args.top_k:
    top_k = args.top_k
if args.json:
    filepath = args.json
if args.gpu:
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
with open(filepath, 'r') as f:
    cat_to_name = json.load(f)

#Process a PIL image for use in a PyTorch model
def process_image(image):
 
    pil_im = Image.open(f'{image}' + '.jpg')

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    
    pil_image = transform(pil_im)
    np_image = np.array(pil_image)
    
    return np_image

#PREDICT USING TRAINED MODEL - taken from https://github.com/paulstancliffe/Udacity-Image-Classifier/blob/master/predict.py
def predict(image_path, model, top_k):

    #Implement the code to predict the class from an image file
    np_image = process_image(image_path)
    
    # Changing from numpy to pytorch tensor
    pytorch_tensor = torch.tensor(np_image)
    pytorch_tensor = pytorch_tensor.float()
    
    # Removing RunTimeError for missing batch size - add batch size of 1 
    pytorch_tensor = pytorch_tensor.unsqueeze(0)
    
    # Run model to make predictions
    model.eval()
    LogSoftmax_predictions = model.forward(pytorch_tensor)
    predictions = torch.exp(LogSoftmax_predictions)
    
    # Identify top predictions and top labels
    top_preds, top_labs = predictions.topk(5)
    
    # Detach top predictions into a numpy list
    top_preds = top_preds.detach().numpy().tolist()
    
    # Change top labels into a list
    top_labs = top_labs.tolist()
    
    # Create a pandas dataframe joining class to flower names
    labels = pd.DataFrame({'class':pd.Series(model.class_to_idx),'flower_name':pd.Series(cat_to_name)})
    labels = labels.set_index('class')
    
    # Limit the dataframe to top labels and add their predictions
    labels = labels.iloc[top_labs[0]]
    labels['predictions'] = top_preds[0]
    
    return labels

model = load_model(checkpoint) 

#Prediction
labels = predict(image_path,model,top_k)
print(labels)
