# %% 
import sys
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree

from PIL import Image, ImageFont, ImageDraw

# path
dataset_path = '/home/vim/Desktop/tykim/workspace/VOC2012'
IMAGE_FOLDER = 'JPEGImages'
ANNOTATIONS_FOLDER = "Annotations"

# os.walk => path / directory / files
ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))
img_root, img_dir, img_files = next(os.walk(os.path.join(dataset_path, IMAGE_FOLDER)))

for xml_file in ann_files[0:5]:

    # parsing image for get it's name
    img_name = img_files[img_files.index(".".join([xml_file.split(".")[0], "jpg"]))]
    img_file = os.path.join(img_root, img_name)
    image = Image.open(img_file).convert("RGB")
    draw = ImageDraw.Draw(image)

    # open xml file that have same name with image
    xml = open(os.path.join(ann_root, xml_file), "r")

    # parsing xml file, using tree stucture
    tree = Et.parse(xml)
    root = tree.getroot()

    size = root.find("size")

    width = size.find("width").text
    height = size.find("height").text
    channels = size.find("depth").text

    objects = root.findall("object")

    for _object in objects:
        name = _object.find("name").text
        bbox = _object.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red")
        draw.text((xmin, ymin), name)
    

    plt.figure(figsize=(25, 20))
    plt.imshow(image)
    plt.show()
    plt.close
print("End.")
# %%
