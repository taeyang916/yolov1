import cv2, os
import random
import numpy as np

import torch
from torch._C import dtype
from torch.utils.data import Dataset
import torchvision.transforms as tfms

class VOC_Dataset(Dataset):
    def __init__(self, image_dir, label_txt, image_size=448, grid_size=7, num_bboxes=2, num_classes=20, train=True, debug=False):
        
        self._train = train
        self._debug = debug
        self.image_size = image_size

        self.S, self.B, self.C = grid_size, num_bboxes, num_classes
        self.mean_rgb = np.array([122.6791434, 116.66876762, 104.00698793], dtype=np.float32)
        self.to_tensor = tfms.Compose([
            tfms.ToTensor(),
            tfms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.paths, self.bboxes, self.labels = [], [], []
        with open(label_txt) as f:
            lines = f.readlines()

        for line in lines:
            attrs = line.strip().split()

            img_name = attrs[0]
            path = os.path.join(image_dir, img_name)
            self.paths.append(path)

            num_bboxes = (len(attrs)-1)/5
            bbox, label = [], []

            for n in range(num_bboxes):
                x1          = float(attrs[5*n + 1])
                y1          = float(attrs[5*n + 2])
                x2          = float(attrs[5*n + 3])
                y2          = float(attrs[5*n + 4])
                confidence  = int(attrs[5*n + 5])
                bbox.append([x1, y1, x2, y2])
                label.append(confidence)

            self.bboxes.append(torch.Tensor(bbox))
            self.labels.append(torch.LongTensor(label))

        self.length = len(self.paths)


    def __getitem__(self, index):
        path = self.paths[index]
        img = cv2.imread(path)
        if self._debug:
            original_img = img.copy()
        
        boxes = self.bboxes[index].clone()
        labels = self.labels[index].clone()

        if self._train:
            img, boxes = self.lr_flip(img, boxes)
            img, boxes = self.ud_flip(img, boxes)
            img, boxes = self.width_scale(img, boxes)
            img, boxes = self.height_scale(img, boxes)
            img = self.blur(img)
            img = self.brightness(img)
            img = self.hue(img)
            img = self.saturation(img)
            img, boxes, labels = self.shift(img, boxes, labels)

        if self._debug:
            debug_dir = './debug'
            os.makedirs(debug_dir, exist_ok=True)

            img_show = img.copy()
            box_show = boxes.numpy().reshape(-1)
            n = len(box_show)

            for b in range(n):
                pt1 = (int(box_show[4*b + 0]), int(box_show[4*b + 1]))
                pt2 = (int(box_show[4*b + 2]), int(box_show[4*b + 3]))
                cv2.rectangle(img_show, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=2)

                cv2.imwrite(os.path.join(debug_dir, f'./train_valid/debug_{index+1}.jpg', img_show))
        
        h, w, _ = img.shape
        boxes /= torch.Tensor([[w,h,w,h]]).expand_as(boxes)
        target = self.to_target(boxes, labels)

        img = cv2.resize(img, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = self.to_tensor(img)
        return img, target

    def __len__(self):
        return self.length

    def to_target(self, bboxes, labels):
        S, B, C = self.S, self.B, self.C
        N = 5 * B + C

        target = torch.zeros(S, S, N)
        cell_size = 1.0/ float(S)
        bboxes_wh = bboxes[:, 2:] - bboxes[:, :2]
        bboxes_ct = (bboxes[:, 2:] + bboxes[:, :2]) / 2.0

        for b in range(bboxes.size(0)):
            ct, wh, label = bboxes_ct[b], bboxes_wh[b], int(labels[b])

            ij = (ct / cell_size).ceil() - 1.0
            i, j = int(ij[0]), int(ij[1])
            base_xy = ij * cell_size
            offset_xy = (ct - base_xy) / cell_size

            for k in range(B):
                s = 5 * k
                target[j, i, s  :s+2] = offset_xy
                target[j, i, s+2:s+4] = wh
                target[j, i, s+4    ] = 1.0
            target[j, i, 5*B + label] = 1.0

        return target

    def lr_flip(self, img, bboxes):
        if random.random() < 0.5:
            return img, bboxes

        h, w, _ = img.shape
        img = np.fliplr(img)

        x1, x2 = bboxes[:, 0], bboxes[:, 2]
        new_x1 = w - x2
        new_x2 = w - x1
        bboxes[:, 0], bboxes[:, 2] = new_x1, new_x2

        return img, bboxes

    def ud_flip(self, img, bboxes):
        if random.random() < 0.5:
            return img, bboxes  

        h, w, _ = img.shape
        img = np.fliplr(img)

        y1, y2 = bboxes[:, 1], bboxes[:, 3]
        new_y1 = h - y2
        new_y2 = h - y1
        bboxes[:, 0], bboxes[:, 2] = new_y1, new_y2    

        return img, bboxes   

    def width_scale(self, img, bboxes):
        if random.random() < 0.5:
            return img, bboxes
        
        scale = random.uniform(0.8, 1.2)
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(int(w*scale), h), interpolation=cv2.INTER_LINEAR)

        scale_tensor = torch.FloatTensor([[scale, 1.0, scale, 1.0]]).expand_as(bboxes)
        bboxes *= scale_tensor

        return img, bboxes

    def height_scale(self, img, bboxes):
        if random.random() < 0.5:
            return img, bboxes
        
        scale = random.uniform(0.8, 1.2)
        h, w, _ = img.shape
        img = cv2.resize(img, dsize=(w, int(h*scale)), interpolation=cv2.INTER_LINEAR)

        scale_tensor = torch.FloatTensor([[ 1.0, scale, 1.0, scale]]).expand_as(bboxes)
        bboxes *= scale_tensor

        return img, bboxes

    def blur(self, img):
        if random.random() < 0.5:
            return img
        
        ksize = random.choice([2, 3, 4, 5])
        img = cv2.blur(img, (ksize, ksize))
        return img

        
