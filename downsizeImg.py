from PIL import Image
import os

#downsize images


def downsizeImg(data_dir,basewidth):
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    for d in directories:
        img_dir = os.path.join(data_dir,d)
        file_name = [os.path.join(img_dir, f)
                     for f in os.listdir(img_dir) if f.endswith(".jpg")]
        for f in file_name:
            img = Image.open(f)
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth,hsize),Image.ANTIALIAS)
            img.save(f)


data_dir = "/home/dingxt/PycharmProjects/BigdataProject/Location-Verfication/frames"
basewidth = 300
downsizeImg(data_dir,basewidth)
