import cv2
import numpy as np
from src.models import Depth,Rotation

from types import SimpleNamespace

class HPE:
    def __init__(self, gpu_id=0, rotation_snapshot=None, depth_model='small'):
        
        data = {'gpu_id': gpu_id, 'snapshot': rotation_snapshot}
        config = SimpleNamespace(**data)

        self.depth_model = Depth(model=depth_model)
        self.rot_model = Rotation(config)

    def process_frame(self, frame):
        bboxes, rotations, rot_image = self.rot_model(frame)
        pred_depth = self.depth_model(frame)

        # taking the first bounding box and rotation for simplicity
        bbox = bboxes[0]
        rotation = rotations[0]

        # get the depth the center of the bounding box
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
    
        depth = pred_depth[center_y, center_x]

        print(depth)
        exit()

        # Assuing a canonical camera model 
        # fx = 1000 and fy = 100
        # cx = image_width / 2
        # cy = image_height / 2

        canonical_camera = np.array([[1000, 0, frame.shape[1] / 2],
                                     [0, 1000, frame.shape[0] / 2],
                                     [0, 0, 1]])
        
        # get the 3D cordinates of the center of the bounding box
        center_2d = np.array([center_x, center_y, 1]) # homogeneous coordinates of the image point of the center

        # convert to 3D coordinates
        # this is the 3D point in the canonical camera coordinate system
        center_3d = np.linalg.inv(canonical_camera) @ center_2d * depth 

        return rotation, center_3d
        