import torch.utils.data as data
import numpy as np
from scipy.misc import imread
import os.path
import random
from pfm_utils import readPFM
from PIL import Image


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/left/0000000.jpg
        root/left/0000001.jpg
        ..

        root/right/0000000.jpg
        ..
        
        root/0000.pfm
        ..
        
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.left_scene_root = 'data/frames_cleanpass_webp/35mm_focallength/scene_forwards/fast/left/'
        self.disp_root = 'data/disparity/35mm_focallength/scene_forwards/fast/left/'
        self.img_names = os.listdir(self.left_scene_root)[:5]
        self.transform = transform
        
    @staticmethod    
    def get_new_path(path, new_dir="right"):
        dir_path, filename = os.path.split(path)
        dir_path, _ = os.path.split(dir_path)
        return os.path.join(dir_path, new_dir, filename)

    def __getitem__(self, index):
        left_img = np.asarray(Image.open(self.left_scene_root + self.img_names[index]).convert("RGB"))
        right_img = np.asarray(Image.open(self.get_new_path(self.left_scene_root) + self.img_names[index]).convert("RGB"))
        disp_map, _ = readPFM(self.disp_root + os.path.splitext(self.img_names[index])[0] + '.pfm')
        if self.transform is not None:
            left_img, right_img, disp_map = self.transform(left_img/255, right_img/255, disp_map)
        else:
            pass #TODO
        return left_img, right_img, disp_map

    def __len__(self):
        return len(self.img_names)