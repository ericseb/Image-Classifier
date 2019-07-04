import numpy as np
import pandas as pd
import seaborn as sb
import json
import time
import copy

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models

from os.path import isdir
from os import listdir
from collections import OrderedDict
import argparse

#SET DEFAULT VALUES
data_dir = 'flowers'
arch = 'vgg19'
hidden_units = 4096
learning_rate = 0.001
epochs = 10
device = 'cpu'
num_output = 102

#ARGPARSE
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir',type=str, help='Location of directory with data for image classifier to train and test')
parser.add_argument('--arch',action='store',type=str, help='Choose pretrained network; vgg19, alexnet, or densenet121. Default = vgg19')
parser.add_argument('--hidden_units',action='store',type=int, help='Number of hidden units. Default = 4096')
parser.add_argument('--learning_rate',action='store',type=float, help='Learning rate for the model. Default = 0.001')
parser.add_argument('--epochs',action='store',type=int, help='Epochs for training model. Default = 10')
parser.add_argument('--save_dir',action='store', type=str, help='File name to save the trained model')
parser.add_argument('--num_output',action='store',type=int, help='Number of image classes to be predicted. Default = 102')
parser.add_argument('--gpu',action='store_true',help='Use GPU if available. Default = cpu')

    
args = parser.parse_args()

if args.data_dir:
    data_dir = args.data_dir
if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.num_output:
    num_output = args.num_output
if args.gpu:        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
#PRETRAINED MODELS 
vgg19 = models.vgg16(pretrained=True)
alexnet = models.alexnet(pretrained=True)
densenet121 = models.densenet121(pretrained=True)

#CREATE MODEL FUNCTION
def create_model(arch, hidden_units, learning_rate):
    # Select from available pretrained models
    model =  getattr(models,arch)(pretrained=True)
   
    model_dict = {"vgg19": vgg19, "alexnet": alexnet, "densenet121": densenet121}
    in_features_dict = {"vgg19": 25088, "alexnet": 9216, "densenet121": 1024}
    
    model = model_dict[arch]
    in_features = in_features_dict[arch]       
    
    #Freeze feature parameters so as not to backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
        
        # Build classifier for model
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, hidden_units)),
                                            ('relu1', nn.ReLU()),
                                            ('dropout1', nn.Dropout(0.5)),
                                            ('fc2', nn.Linear(hidden_units, num_output)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier   
            
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.1,last_epoch=-1)
    
    return model, criterion, optimizer, scheduler

model, criterion, optimizer, scheduler = create_model(arch, hidden_units, learning_rate)

print("-" * 10)
print("Your model has been built!")

# Set Image Directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       }

#Transform Data & Create Data Loaders
data_dict = {'train': train_dir,
             'valid': valid_dir}

image_datasets = {x: datasets.ImageFolder(data_dict[x],transform = data_transforms[x])
    for x in ['train', 'valid']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,shuffle=True) 
    for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

#MODEL TRAINING
#train_model taken from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data

def train_model(model, criterion, optimizer, scheduler, epochs):
    since = time.time()
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} | {} Accuracy: {:.4f}'.format(
                phase, epoch_loss, phase, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_trained = train_model(model, criterion, optimizer, scheduler, epochs)

print('-' * 10)
print('Training Complete!!, Thanks for waiting')
print('-' * 10)

#SAVING TRAINED MODEL
def save_model(model_trained):

    model_trained.class_to_idx = image_datasets['train'].class_to_idx
    model_trained.cpu()
    save_dir = ''
    checkpoint = {
             'arch': arch,
             'num_output': num_output,
             'hidden_units': hidden_units, 
             'state_dict': model_trained.state_dict(),
             'class_to_idx': model_trained.class_to_idx,
             }
    
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'checkpoint.pth'

    torch.save(checkpoint, save_dir)    
    
save_model(model_trained)
print('-' * 10)
print(model_trained)
print('Your trained model has been successfully saved.')
print('-' * 10)
