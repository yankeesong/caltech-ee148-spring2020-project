from __future__ import print_function, division
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import matplotlib.ticker as mtick
from scipy.integrate import cumtrapz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io, transform, color, img_as_float64
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler, WeightedRandomSampler
import time




### Add integer label to the dataframe
def add_label(df, df1,df2,df3):

    '''
    Attach a new column named Label to the dataframe:
    integer labels of the images according to their classes (0 - 782);

    Args:
        df (pandas.Dataframe): the original df without int labels
    Return:
        df (pandas.Dataframe): the new df with int labels
    '''
    classes = df['scientific_name']
    labels1 = []
    labels2 = []
    labels3 = []
    remove1 = []
    remove2 = []
    remove3 = []
    data_address = "../snake-species-identification-challenge/data/"
    print("start df3")
    found = 0
    for i in range(len(df3)):
        if i % 10 == 0:
            print(i)
        img_name = os.path.join("../snake-species-identification-challenge/data/validate_images_small/",(df3['hashed_id'][i]+'.jpg'))  
        img = io.imread(img_name)
        class_name = df3['scientific_name'][i]
        if len(img.shape) == 2:
            found += 1
            df3 = df3.drop(df3.index[i])
        labels3.append(classes[classes == class_name].index[0])
    print(found)
    df3['Label'] = labels3
    df3.to_csv(os.path.join(data_address,'processed_validate_labels_small.csv'))
    
    print("start df2")
    found = 0
    for i in range(len(df2)):
        if i % 1000 == 0:
            print(i)
        img_name = os.path.join("../snake-species-identification-challenge/data/validate_images/",(df2['hashed_id'][i]+'.jpg'))  
        img = io.imread(img_name)
        class_name = df2['scientific_name'][i]
        if len(img.shape) == 2:
            found += 1
            remove2.append(i)
        labels2.append(classes[classes == class_name].index[0])
    print(found)
    df2['Label'] = labels2
    df2 = df2.drop(remove2)
    df2.to_csv(os.path.join(data_address,'processed_validate_labels.csv'))
        
    print("start df1")
    found = 0
    for i in range(len(df1)):
        if i % 5000 == 0:
            print(i)
        img_name = os.path.join("../snake-species-identification-challenge/data/train_images/",(df1['hashed_id'][i]+'.jpg'))  
        img = io.imread(img_name)
        class_name = df1['scientific_name'][i]
        if len(img.shape) == 2:
            found += 1
            remove1.append(i)
        labels1.append(classes[classes == class_name].index[0])
    print(found)
    df1['Label'] = labels1
    df1 = df1.drop(remove1)
    df1.to_csv(os.path.join(data_address,'processed_train_labels.csv'))

### Preprocessing
class preprocessing():
    def __init__(self, resize = False, size=(224,224)):
        self.resize = resize  # if resize or not
        self.size = size # Resize
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])   # Add more steps to achieve augmentation

    def new_image(self,image):
        if self.resize:
            image = transform.resize(image,self.size)
        else:
            # Transform Uint8 into float64
            image = img_as_float64(image)
        image = self.transform(image)
        return image


### Create Custom Dataset (Old)
class OldSnakeDataSet(Dataset):
    def __init__(self, filename, image_path, preprocess=preprocessing()):
        '''
        Args:
            filename (str): filename of the csv that stores the pd.dataframe with the int labels
            image_path (str): the path of the folder where images are stored
            preprocess (bool): Whether to preprocess on image
        '''
        self.data = pd.read_csv(filename,dtype={'hashed_id':str})    # The
        self.id = self.data['hashed_id']       # The hashed ids of the images
        self.labels = self.data['Label']     # The int labels of the images
        self.path = image_path               # The directory the images are
        self.preprocess = preprocess           # The transform performed on the images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #print(idx)
        print(self.id[idx])
        image = io.imread(os.path.join(self.path,(self.id[idx]+'.jpg')))
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        label = self.labels[idx]
        if self.preprocess:
            image = self.preprocess.new_image(image)
        sample = (image, label)
        return sample
    
  

### Create Custom Dataset 
class SnakeDataSet(Dataset):
    def __init__(self, filename, image_path, preprocess=preprocessing(size=(250,250)), resize = False):
        '''
        Args:
            filename (str): filename of the csv that stores the pd.dataframe with the int labels
            image_path (str): the path of the folder where images are stored
            preprocess (bool): Whether to preprocess on image
            resize (bool): Whether to resize the image when read it
        '''
        self.data = pd.read_csv(filename)    # The
        self.id = self.data['hashed_id']       # The hashed ids of the images
        self.labels = self.data['Label']     # The int labels of the images
        self.path = image_path               # The directory the images are
        self.preprocess = preprocess          # The transform performed on the images
        self.resize = resize                   # Whether resize or not

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(self.path,(self.id[idx]+'.jpg')))
        if self.resize:           # If using raw training images
            if image.shape[2] == 4:
                image = color.rgba2rgb(image)
            if self.preprocess:
                image = self.preprocess.new_image(image)
        else:               # If using resized images, no need to resize again
            if self.preprocess:
                image = self.preprocess.new_image(image)

        label = self.labels[idx]
        sample = (image, label)
        return sample



### CNN class
class BasicNet(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(BasicNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(6,6), stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), stride=1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2312, 1500)
        self.fc2 = nn.Linear(1500, 783)
        self.bn = nn.BatchNorm2d(8)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout1(x)
        output = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output
    
    
### AlexNet
class AlexNet(nn.Module):

    def __init__(self, num_classes=783):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

    
def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        #input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        #input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        #input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        #input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        #input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        #input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
    
### Function to train the Net
def train(model, device, train_loader, fraction, optimizer, epoch, log_interval = 100):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    train_length = int(len(train_loader.dataset) / fraction)
    model.train()   # Set the model to training mode
    ts = time.time()  # Starting time
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data.float())                # Make predictions
        #loss = F.nll_loss(output, target)   # Compute loss
        lossf = nn.CrossEntropyLoss()
        loss = lossf(output,target)
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step

        if batch_idx % log_interval == 0:
    
            # Report each 100 batch
            f=open('running.txt','a')
            message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), train_length,
                100. * batch_idx *len(data) / train_length, loss.item())
            f.write(message)
            print(message)
            te = time.time()                             # ending time of 100 batchs
            message = 'Time Elapsed: %.i s.\n' % (int(te - ts))
            f.write(message)
            print(message)
            f.close()
            ts = time.time()                             # restart Timer
            

### Test function
def test(model, device, test_loader, fraction):

    correct = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data0, target0 in test_loader:
            data, target = data0.to(device), target0.to(device)
            output = model(data.float())  
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    
    test_length = int(len(test_loader.dataset) / fraction)
    test_loss /= test_length

    # Report each epoch
    f = open('running.txt', 'a')
    message = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_length,
        100. * correct / test_length)
    f.write(message)
    f.close()
    print(message)
        
    return test_loss

