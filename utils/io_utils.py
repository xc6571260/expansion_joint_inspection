import cv2
import os

def load_image(path):
    if not os.path.exists(path):
        return None
    return cv2.imread(path)

def save_image(img, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
