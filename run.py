import cv2
import numpy as np
from src.main import HPE

cap = cv2.VideoCapture(0)

hpe = HPE(rotation_snapshot='./src/models/rotation/pre/cmu.pth', depth_model='small')

while True:

    ret,frame = cap.read()

    if not ret:
        break

    rot, t = hpe.process_frame(frame.copy())

    print(f'Rotation : {rot}\nTranslation: {t}')

    cv2.imshow('Map', frame)
    c = cv2.waitKey(1) &0xFF

    if c==ord('q'):
        break 