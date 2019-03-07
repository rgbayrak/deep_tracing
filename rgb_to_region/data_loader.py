'''
This class organizes the dataset in a way that is useful for learning.
It takes an RGB input image, seed, ROI, ROA regions as target images,
get them ready to send in to Pytorch's dataloader.
'''
import numpy as np
from torch.utils.data import Dataset
from utils import *
import nibabel as nib
import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

'''
'''
class dataset(Dataset):
    # inherited from torch's Dataset
    def __init__(self, paths):
        self.data = self.input_target(paths)
        print(len(self.data))

    def __len__(self):
        return len(self.data)  # number of images (training input, validation input)

    def init_dict(self, su, ra, input_file):
        d = {}
        d['subject'] = []
        d['subject'].append(su)
        d['rater'] = []
        d['rater'].append(ra)
        d['rgb'] = []
        d['rgb'].append(input_file)
        d['seeds'] = []
        d['rois'] = []
        d['roas'] = []

        return d

    def input_target(self, paths):

        data_files = [] # create a list

        # for i in ['/share4/bayrakrg/tractEM/postprocessing/rgb_files/HCP', '/share4/bayrakrg/tractEM/postprocessing/rgb_files/BLSA']:
        for i in paths:
        # ////////////////////////////
        # HCP structure to dictionary
        # ////////////////////////////
            if 'HCP' in i:
                files = os.listdir(i)
                files.sort()
                for f in files:
                    if 'rgb.nii' in f:
                        subj = f.split('_')[0]
                        input_file = os.path.join(i, f)
                        target_path = '/home-local/bayrakrg/Dropbox (VUMC)/complete_corrected_HCP_subjects'
                        subject = os.listdir(target_path)
                        subject.sort()
                        for s in subject:
                            su = s.split('_')[0]
                            ra = s.split('_')[1]
                            if subj in s:
                                tract = os.listdir(os.path.join(target_path, s))
                                tract.sort()
                                for tr in tract:
                                    if tr == "genu_corpus_callosum":
                                        targetDir = os.path.join(target_path, s, tr)
                                        target_files = os.listdir(targetDir)

                                        d = self.init_dict(su, ra, input_file)
                                        dleft = self.init_dict(su, ra, input_file)
                                        bilat = False
                                        for t in target_files:
                                            if 'seed' in t:
                                                if '_L_' in t:
                                                    bilat = True
                                                    dleft['seeds'].append(os.path.join(targetDir,t))
                                                elif '_R_' in t:
                                                    d['seeds'].append(os.path.join(targetDir, t))
                                                else:
                                                    d['seeds'].append(os.path.join(targetDir, t)) # not the right
                                                    dleft['seeds'].append(os.path.join(targetDir, t))
                                            if 'ROI' in t:
                                                if '_L_' in t:
                                                    bilat = True
                                                    dleft['rois'].append(os.path.join(targetDir,t))
                                                elif '_R_' in t:
                                                    d['rois'].append(os.path.join(targetDir, t))
                                                else:
                                                    d['rois'].append(os.path.join(targetDir, t))
                                                    dleft['rois'].append(os.path.join(targetDir, t))
                                            if 'ROA' in t:
                                                d['roas'].append(os.path.join(targetDir, t))
                                                dleft['roas'].append(os.path.join(targetDir, t))

                                        data_files.append(d)
                                        if bilat:
                                            data_files.append(dleft)

            # ///////////////////////////////
            # BLSA structure to dictionary
            # ///////////////////////////////
            if 'BLSA' in i:
                files = os.listdir(i)
                files.sort()
                for f in files:
                    if 'rgb.nii' in f:
                        subj = f.split('_')[0]
                        input_file = os.path.join(i, f)
                        target_path = '/home-local/bayrakrg/Dropbox (VUMC)/complete_corrected_BLSA_subjects'
                        subject = os.listdir(target_path)
                        subject.sort()
                        for s in subject:
                            if s != 'postproc':
                                su = s.split('_')[0]
                                ra = s.split('_')[1]
                                if subj in s:
                                    tract = os.listdir(os.path.join(target_path, s))
                                    tract.sort()
                                    for tr in tract:
                                        if tr == "genu_corpus_callosum":
                                            targetDir = os.path.join(target_path, s, tr)
                                            target_files = os.listdir(targetDir)

                                            d = self.init_dict(su, ra, input_file)
                                            dleft = self.init_dict(su, ra, input_file)
                                            bilat = False
                                            for t in target_files:
                                                if 'seed' in t:
                                                    if '_L_' in t:
                                                        bilat = True
                                                        dleft['seeds'].append(os.path.join(targetDir,t))
                                                    elif '_R_' in t:
                                                        d['seeds'].append(os.path.join(targetDir, t))
                                                    else:
                                                        d['seeds'].append(os.path.join(targetDir, t)) # not the right
                                                        dleft['seeds'].append(os.path.join(targetDir, t))
                                                if 'ROI' in t:
                                                    if '_L_' in t:
                                                        bilat = True
                                                        dleft['rois'].append(os.path.join(targetDir,t))
                                                    elif '_R_' in t:
                                                        d['rois'].append(os.path.join(targetDir, t))
                                                    else:
                                                        d['rois'].append(os.path.join(targetDir, t))
                                                        dleft['rois'].append(os.path.join(targetDir, t))
                                                if 'ROA' in t:
                                                    d['roas'].append(os.path.join(targetDir, t))
                                                    dleft['roas'].append(os.path.join(targetDir, t))

                                            data_files.append(d)
                                            if bilat:
                                                data_files.append(dleft)

            # # ///////////////////////////////
            # # BLSA19 structure to dictionary
            # # ///////////////////////////////
            # if 'BLSA19' in i:
            #     files = os.listdir(i)
            #     files.sort()
            #     for f in files:
            #         if 'rgb.nii' in f:
            #             subj = f.split('_')[0]
            #             input_file = os.path.join(i, f)
            #             target_path = '/home-local/bayrakrg/Dropbox (VUMC)/19_new_BLSA'
            #             subject = os.listdir(target_path)
            #             subject.sort()
            #             for s in subject:
            #                 if s != '.dropbox':
            #                     su = s.split('_')[0]
            #                     ra = s.split('_')[1]
            #                     if subj in s:
            #                         tract = os.listdir(os.path.join(target_path, s))
            #                         tract.sort()
            #                         for tr in tract:
            #                             if tr == "genu_corpus_callosum":
            #                                 targetDir = os.path.join(target_path, s, tr)
            #                                 target_files = os.listdir(targetDir)
            #
            #                                 d = self.init_dict(su, ra, input_file)
            #                                 dleft = self.init_dict(su, ra, input_file)
            #                                 bilat = False
            #                                 for t in target_files:
            #                                     if 'seed' in t:
            #                                         if '_L_' in t:
            #                                             bilat = True
            #                                             dleft['seeds'].append(os.path.join(targetDir, t))
            #                                         elif '_R_' in t:
            #                                             d['seeds'].append(os.path.join(targetDir, t))
            #                                         else:
            #                                             d['seeds'].append(
            #                                                 os.path.join(targetDir, t))  # not the right
            #                                             dleft['seeds'].append(os.path.join(targetDir, t))
            #                                     if 'ROI' in t:
            #                                         if '_L_' in t:
            #                                             bilat = True
            #                                             dleft['rois'].append(os.path.join(targetDir, t))
            #                                         elif '_R_' in t:
            #                                             d['rois'].append(os.path.join(targetDir, t))
            #                                         else:
            #                                             d['rois'].append(os.path.join(targetDir, t))
            #                                             dleft['rois'].append(os.path.join(targetDir, t))
            #                                     if 'ROA' in t:
            #                                         d['roas'].append(os.path.join(targetDir, t))
            #                                         dleft['roas'].append(os.path.join(targetDir, t))
            #
            #                                 data_files.append(d)
            #                                 if bilat:
            #                                     data_files.append(dleft)

        return data_files

    def load_image(self, n):
        input_file = self.data[n]['rgb']
        input_img = nib.load(input_file[0]).get_fdata()

        if len(self.data[n]['seeds']) > 0 and len(self.data[n]['roas']) > 0: # make sure there is at least one seed one ROA
            # load all the labels into one volume as onehot
            target_img = np.zeros([input_img.shape[0], input_img.shape[1], input_img.shape[2], 2]) # xyz of the image and 3 channels for the labels w/o background

            # background
            target_img[:, :, :, 0] = np.ones([input_img.shape[0], input_img.shape[1], input_img.shape[2]])

            # seed
            for f in range(len(self.data[n]['seeds'])):
                target_file = self.data[n]['seeds'][0]
                seed = nib.load(self.data[n]['seeds'][f]).get_fdata()
                target_img[:, :, :, 1] = np.add(seed, target_img[:, :, :, 1])
                target_img[:, :, :, 0] = target_img[:, :, :, 0] - target_img[:, :, :, 1]

            # # roi
            # for u in range(len(self.data[n]['rois'])):
            #     roi = nib.load(self.data[n]['rois'][u]).get_fdata()
            #     target_img[:, :, :, 2] = np.add(roi, target_img[:, :, :, 2])
            #     target_img[:, :, :, 0] -= target_img[:, :, :, 2]

            # # roa
            # roa = nib.load(self.data[n]['roas'][0]).get_fdata()
            # target_img[:, :, :, 3] = roa
            # target_img[:, :, :, 0] -= target_img[:, :, :, 3]

            # flip the image upside down
            target_img = np.flip(target_img, axis=1)
            # print(input_file)
            # print(self.data[n])
            return input_img, target_img, input_file[0]
        else:
            print("Inadequate amount of labels! " % input_file)
            # del self.data[n] #removed it if there is no seeds or roas
            # n = min(n, len(self.data))
            # return self.load_image(n) #call the function skipping this item


    def __getitem__(self, idx):

        input, target, input_file = self.load_image(idx)
        # input = (input - input.min()) / (input.max() - input.min())

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input = input.transpose((3, 0, 1, 2)) # xyz channel --> channel xzy
        target = target.transpose((3, 0, 1, 2))

        input = input.astype(np.float32)
        target = target.astype(np.float32)

        # # Change image size here
        input = F.interpolate(torch.from_numpy(input).unsqueeze(0),(112,112,112), mode='trilinear', align_corners=True).squeeze()
        target = F.interpolate(torch.from_numpy(target).unsqueeze(0), (112, 112, 112), mode='nearest').squeeze()

        save_path = '/share4/bayrakrg/tractEM/postprocessing/deep_tracing/nifti/' + input_file[-17:-11] + str(idx) + '_input.nii'
        save_nifti(input, input_file, save_path)
        # save_path2 = '/share4/bayrakrg/tractEM/postprocessing/deep_tracing/nifti/' + input_file[
        #                                                                             -17:-11] + idx + '_target.nii'
        # save_nifti(input, input_file, save_path2)
        #
        return {'input': input, 'target': target, 'location': input_file}

# # #for testing comment out at runtime
# D = dataset(['/share4/bayrakrg/tractEM/postprocessing/rgb_files/HCP'])
# D[0]




