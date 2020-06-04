import pandas as pd
import numpy as np
import os
from skimage import io, transform, color, img_as_ubyte
from PIL import Image
from torchvision import transforms, utils, datasets
import time
import sys

df = pd.read_csv("processed_train_labels.csv", dtype={'hashed_id':str})
df.sort_values(by='scientific_name',inplace=True)
count = df['scientific_name'].value_counts().sort_index()
new_df = pd.DataFrame()          #generate new csv for dataset
augment_picture = pd.DataFrame()    #keep track of augemented images
image_path = "resized_train_images_250"  # Path storing original images
save_path = image_path # Path to save aug images
level = 500     # Determine whether to do data augmentation or random selection
augmentation_set = [transforms.RandomRotation(180), transforms.RandomAffine(180, translate=(0.2,0.2)),
                    transforms.RandomResizedCrop(size=(250,250),scale=(0.5, 0.9)), transforms.RandomVerticalFlip(p=1.0),
                    transforms.RandomHorizontalFlip(p=1.0),transforms.RandomPerspective(distortion_scale=0.2,p=1.0)] # candidate transformations


for i in range(783):
    # Get the ensemble of one class
    df_slice = df.iloc[sum(count[0:i]):sum(count[0:i+1]),:]
    # if this class have more than level (500) data, random sample 500
    if count[i] >= level:
        new_df = new_df.append(df_slice.sample(n=level))
        continue

    # if this class have fewer than 500 data, perform data augmentation
    for j in range(level-count[i]):
        sample = df_slice.sample(n=1)              #Randomly choose the image to be transformed
        sample_id = sample.iloc[0]['hashed_id']
        aug_id = sample_id + str(j)                #Generate the augmented image id (the original image name + the loop order)
        sample['hashed_id'] = aug_id               #Record the generated image id and information
        augment_picture = augment_picture.append(sample)


        image = transforms.Resize(size=(250,250)) (Image.open(os.path.join(image_path, sample_id+'.jpg')))        #Open the original image
        aug_image = transforms.RandomChoice(augmentation_set)(image)                       #Perform a random transformation from the transformation set
        aug_image.save(os.path.join(save_path,aug_id+'.jpg'))

    new_df = new_df.append(df_slice)
    print("Class %i; Time: %.2f"%(i, time.time()))
new_df = new_df.append(augment_picture)
augment_picture.to_csv("augmented_images.csv")
new_df.to_csv("augmented_train_labels.csv")
print("Data Augmentation Finished.")


