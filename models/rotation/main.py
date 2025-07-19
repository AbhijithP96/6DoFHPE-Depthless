import time
import math
import re
import sys
import os
import argparse
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
from .model import RepNet6D, RepNet5D
from . import utils
import cv2
from .demo.hdssd import Head_detection
import tensorflow as tf
from pprint import pprint


class Rotation:
    def __init__(self,config):
        
        self.head_detect = Head_detection('./models/rotation/demo/SSD_models/Head_detection_300x300.pb')
        self.transform = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        cudnn.enabled = True
        gpu = config.gpu_id
        if (gpu < 0):
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:%d' % gpu)
        # cam = args.cam_id
        snapshot_path = config.snapshot
        self.model = RepNet6D(backbone_name='RepVGG-B1g4',
                        backbone_file='',
                        deploy=True,
                        pretrained=False)

        print('Initializing HPE model.')

        print('model', self.model)

        # Load snapshot
        saved_state_dict = torch.load(os.path.join(snapshot_path), map_location=None if torch.cuda.is_available() else 'cpu')

        if 'model_state_dict' in saved_state_dict:
            self.model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(saved_state_dict)
        self.model.to(device)
        self.model.eval()

        self.device = device

    def __call__(self, image):

        h,w = image.shape[:2]
        image, heads = self.head_detect.run(image, w, h)
        #heads = heads[0] # taking only head, comment to include all heads
        rotations = []
        
        for dict in heads:
            x_min = int(dict['left'])
            y_min = int(dict['top'])
            x_max = int(dict['right'])
            y_max = int(dict['bottom'])
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)

            x_min = max(0, x_min-int(0.2*bbox_height))
            y_min = max(0, y_min-int(0.2*bbox_width))
            x_max = x_max+int(0.2*bbox_height)
            y_max = y_max+int(0.2*bbox_width)

            img = image[y_min:y_max, x_min:x_max]
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = self.transform(img)
            img = torch.Tensor(img[None, :]).to(self.device)

            start = time.time()
            R_pred = self.model(img)
            end = time.time()
            #print('Head pose estimation: %2f ms' % ((end - start)*1000.))

            rotations.append(R_pred.detach().cpu().numpy()[0])

            euler = utils.compute_euler_angles_from_rotation_matrices(R_pred)*180/np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            utils.draw_axis(image, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], x_min + int(.5*(
                x_max-x_min)), y_min + int(.5*(y_max-y_min)), size=bbox_width)
            
        return rotations, image 
    


if __name__ == '__main__':

    from types import SimpleNamespace

    data = {'gpu_id': 0, 'snapshot': './pre/300W_LP.pth'}
    config = SimpleNamespace(**data)

    rot = Rotation(config)

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rotations, image = rot(frame)
        print(rotations)

        cv2.imshow('pred', image)
        c = cv2.waitKey(1) &0xFF

        if c== ord('q'):
            break
