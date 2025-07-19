import cv2
import numpy as np
from models import Depth,Rotation

from types import SimpleNamespace

data = {'gpu_id': 0, 'snapshot': './models/rotation/pre/300W_LP.pth'}
config = SimpleNamespace(**data)

depth_model = Depth()
rot_model = Rotation(config)
cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()

    if not ret:
        break

    pred_depth = depth_model(frame)
    pred_rot, rot_image = rot_model(frame)

    depth_map = pred_depth.detach().cpu().numpy()
    depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    depth_map = np.uint8(depth_map*255)
    depth_map = cv2.applyColorMap(255-depth_map, cv2.COLORMAP_JET)

    cv2.imshow('Map', depth_map)
    cv2.imshow('Rot', rot_image)
    c = cv2.waitKey(1) &0xFF

    if c==ord('q'):
        break 