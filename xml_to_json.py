import json
from itertools import chain
import xmltodict
import os
from xml.etree.ElementTree import parse

# path
dataset_path = '/home/vim/Desktop/tykim/workspace/VOC2012'
IMAGE_FOLDER = 'JPEGImages'
ANNOTATIONS_FOLDER = "Annotations"

json_list = []
ann_root, ann_dir, ann_files = next(os.walk(os.path.join(dataset_path, ANNOTATIONS_FOLDER)))

for xml_file in ann_files:
    xml_ = open(os.path.join(ann_root, xml_file), "r")
    xmlString = xmltodict.parse(xml_.read())
    parsed_xml = xmlString["annotation"]
    json_list.append(parsed_xml)

# for (i, xml_file) in enumerate(ann_files):        
#     xml_ = open(f"/home/vim/Desktop/tykim/workspace/VOC2012/json/{xml_file}.json", "r")
#     xmlString = xmltodict.parse(xml_.read())
#     parsed_xml = xmlString["annotation"]
    # globals()['json_'+str(i)] = parsed_xml

# for (i, dt) in enumerate(ann_files):
#     json_list.append(globals()['json_'+str(i)])
print(json_list[0])
parsed_json_list = [{"annotations", json_list}]

print(parsed_json_list[0])
# slack = list(chain.from_iterable(json_list["annotation"]))
# print(slack)

# with open("/home/vim/Desktop/tykim/workspace/VOC2012/json/annotations.json", 'w') as ann:
#     json.dump(json_list, ann, indent=4)