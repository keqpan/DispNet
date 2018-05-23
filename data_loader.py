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

    def __init__(self, filenames, seed=None, transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.img_names = filenames
        self.transform = transform
        
        self.disp7_size = (15, 9)
        self.disp6_size = (30, 17)
        self.disp5_size = (60, 34)
        self.disp4_size = (120, 68)
        self.disp3_size = (240, 135)
        self.disp2_size = (480, 270)
        
    @staticmethod    
    def get_new_path(path, new_dir="right"):
        dir_path, filename = os.path.split(path)
        dir_path, _ = os.path.split(dir_path)
        return os.path.join(dir_path, new_dir, filename)
    
    @staticmethod
    def get_disp_path(path):
        tmp_arr = path.split('/', 2)
        tmp_arr[1] = "disparity"
        return "/".join(tmp_arr)

    def __getitem__(self, index):
        left_img = np.asarray(Image.open(self.img_names[index]).convert("RGB"))
        right_img = np.asarray(Image.open(self.get_new_path(self.img_names[index])).convert("RGB"))
        disp_map, _ = readPFM(os.path.splitext(self.get_disp_path(self.img_names[index]))[0] + '.pfm')
<<<<<<< HEAD
        disp_img_tmp = Image.fromarray(disp_map)
        
=======
>>>>>>> 56370ebd6e295dbc9cf13a4b8d1b7ce816ebe3ea
        if self.transform is not None:
            left_img, right_img, disp_map = self.transform(left_img/255, right_img/255, disp_map)
        else:
            pass #TODO

        disp7 = np.asarray(disp_img_tmp.resize(self.disp7_size))
        disp6 = np.asarray(disp_img_tmp.resize(self.disp6_size))
        disp5 = np.asarray(disp_img_tmp.resize(self.disp5_size))
        disp4 = np.asarray(disp_img_tmp.resize(self.disp4_size))
        disp3 = np.asarray(disp_img_tmp.resize(self.disp3_size)) 
        disp2 = np.asarray(disp_img_tmp.resize(self.disp2_size))

        return left_img, right_img, (disp_map, disp2, disp3, disp4, disp5, disp6, disp7)

    def __len__(self):
        return len(self.img_names)