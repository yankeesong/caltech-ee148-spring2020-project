import os
import time

image_path = "resized_train_images_250"
os.chdir(image_path)
filenames = sorted(os.listdir("."))
print(len(filenames))
delete_count = 0 
for i in range(len(filenames)):
    if len(filenames[i])>14:
        os.system("rm -f " + filenames[i])
        delete_count += 1
    if i % 10000 == 0:
        print(i, time.time(), delete_count, len(os.listdir(".")))
print("Revoke Finished.")
