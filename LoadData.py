import os
import skimage.data
import skimage.transform
import skimage.data
from skimage import io
from skimage import color

def load_data(data_dir):
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    #print(directories)

    labels = []
    images = []

    for d in directories:
        label_dir = os.path.join(data_dir, d)
        #print(label_dir)
        file_name = [os.path.join(label_dir, f)
                     for f in os.listdir(label_dir) if f.endswith(".jpg")]
        #print(len(file_name))
        for f in file_name:
            #img = skimage.data.imread(f)
            images.append(skimage.data.imread(f))
            #images.append(color.rgb2gray(io.imread(f)))
            labels.append(int(d))
        #print(len(images))
    return images ,labels

#training_data_dir = '/home/parallels/BigdataProject/frames/'
#images,labels = load_data(training_data_dir)
#print(labels)


