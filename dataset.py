import math
import argparse

import torchvision
from torchvision.models import vgg16, vgg16_bn
from torchsummaryM import summary
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from model import *
from loss import YOLO_loss
from utils import *




dataset = VOC_Dataset(args.img, args.label, debug=args.debug)