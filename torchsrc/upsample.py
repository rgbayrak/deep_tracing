
import nibabel as nib
import numpy as np
import os
from skimage import transform

img = nib.load('/home-local/bayrakrg/MDL/Ass3_Segmentation/assignment3/Testing/img/img0080.nii.gz')
size = img.shape
size = [size[0], size[1], size[2]]
label = nib.load('/home-local/bayrakrg/MDL/Ass3_Segmentation/v3D/SLANTbrainSeg-master/latest/seg_output/seg/label0080.nii.gz')
label = label.get_data()
label = np.transpose(label, (1, 2, 0))
label = transform.resize(label, size, order=0 , mode='constant', anti_aliasing=False)
out_nii_file = os.path.join('/home-local/bayrakrg/MDL/Ass3_Segmentation/v3D/SLANTbrainSeg-master/latest/seg_output/upsampled_seg/label0080.nii.gz')
label = nib.Nifti1Image(label, affine=np.eye(4))
nib.save(label, out_nii_file)
