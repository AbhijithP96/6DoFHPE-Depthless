import cv2
import numpy as np
import torch

from .hubconf import metric3d_vit_small,metric3d_vit_large,metric3d_vit_giant2

class Depth:
    def __init__(self, model= 'small'):
        models = {'small' : metric3d_vit_small,
                  'large' : metric3d_vit_large,
                  'giant' : metric3d_vit_giant2}
        
        self.model = models[model](pretrain=True).cuda().eval()
        
        self.mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        self.std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

        self.input_size = (616, 1064) # for vit model

    def __call__(self, image, intrinsic= None):
        image = image[:, :, ::-1]
        h,w = image.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        rgb = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        if intrinsic is None:
            intrinsic = [1000.0, 1000.0, image.shape[1]/2, image.shape[0]/2]

        intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = self.input_size[0] - h
        pad_w = self.input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        #### normalize
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - self.mean), self.std)
        rgb = rgb[None, :, :, :].cuda()

        ###################### canonical camera space ######################
        # inference
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.model.inference({'input': rgb})

        # un pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
        
        # upsample to original size
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], image.shape[:2], mode='bilinear').squeeze()
        ###################### canonical camera space ######################

        #### de-canonical transform
        canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300)

        return pred_depth.detach().cpu().numpy()


if __name__ == '__main__':

    depth = Depth()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        pred_depth = depth(frame)

        depth_map = pred_depth.detach().cpu().numpy()
        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        depth_map = np.uint8(depth_map*255)
        depth_map = cv2.applyColorMap(255-depth_map, cv2.COLORMAP_JET)

        cv2.imshow('Map', depth_map)
        c = cv2.waitKey(1) &0xFF

        if c==ord('q'):
            break 

