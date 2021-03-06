import os
import numpy as np
from torch.utils import data
import nibabel as nib
from skimage import transform
from nilearn import plotting

nRows = 128
nCols = 128
nSlices = 64

output_x = 96
output_y = 96
output_z = 88


labels = [0, 1] # to be changed
class pytorch_loader(data.Dataset):
    def __init__(self, dict, num_labels):
        self.dict = dict
        self.keys = list(dict.keys())
        self.num_labels = num_labels

    def __getitem__(self, index):
        num_labels = self.num_labels
        img_file = self.dict[self.keys[index]]
        sub_name = self.keys[index].split('/')[-3]
        img_3d = nib.load(img_file)
        img = np.array(img_3d.get_data())

        # new_shape = [output_x, output_y, output_z]
        # img = transform.resize(img, new_shape, mode='constant', preserve_range='True', anti_aliasing=True)
        img = np.transpose(img, (3, 0, 1, 2))
        img = np.expand_dims(img, axis=0)
        x = img
        x = (x - x.min())/(x.max()-x.min())
        x = x.astype('float32')


        y = np.zeros((num_labels, output_z, output_x, output_y))
        seg_file = self.keys[index]
        seg_3d = nib.load(seg_file)
        seg = seg_3d.get_data()

        # new_shape = [output_x, output_y, output_z]
        # mask = np.array([seg == 1][0]) # create a binary segmentation mask
        # mask = transform.resize(mask, new_shape, order=0, mode='constant', preserve_range='True', anti_aliasing=False)

        seg = np.transpose(seg, (2, 0, 1))
        y[0,:,:,:] = np.ones([output_z,output_x,output_y]) # create an array of same size
        for i in range(1,num_labels):
            y[i,:,:,:] = seg[0:output_z,0:output_x,0:output_y]
            y[0,:,:,:] = y[0,:,:,:] - y[i,:,:,:]
        y = y.astype('float32')

        return x, y, sub_name

    def __len__(self):
        return len(self.keys)