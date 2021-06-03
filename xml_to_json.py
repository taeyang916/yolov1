import json
from itertools import chain
import xmltodict
import os
from xml.etree.ElementTree import parse
# path
dataset_path = '/home/vim/Desktop/tykim/workspace/VOC2012'
IMAGE_FOLDER = 'JPEGImages'
ANNOTATIONS_FOLDER = "Annotations"

ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))
for xml_file in ann_files:
    xml_ = open(os.path.join(ann_root, xml_file), "r")
    xmlString = xml_.read()
    jsonString = json.dumps(xmltodict.parse(xmlString), indent=4)
 

for (i, xml_file) in enumerate(ann_files):        
        xml_ = open(f"/home/vim/Desktop/tykim/workspace/VOC2012/json/{xml_file}.json", "r")
        globals()['json_'+str(i)] = json.load(xml_)

json_list = []
for (i, dt) in enumerate(ann_files):
    json_list.append(globals()['json_'+str(i)])
print(json_list[5])
slack = list(chain.from_iterable(json_list))

with open("/home/vim/Desktop/tykim/workspace/VOC2012/json/annotations.json", 'w') as ann:
    json.dumps(slack, indent=4)