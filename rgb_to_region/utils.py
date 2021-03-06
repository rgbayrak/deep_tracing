from __future__ import print_function, division
import os
import torch

import numpy as np
from scipy import interpolate
import nibabel as nib
import warnings
# warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import torch.nn.functional as F

def save_nifti(batch_img, path_orig_nifti, save_path):
    batch_img = batch_img.detach().cpu().numpy()
    if batch_img.ndim > 4:
        img = batch_img[0, :, :, :, :]
    else:
        img = batch_img
    img = np.squeeze(img)

    img = F.interpolate(torch.from_numpy(img).unsqueeze(0), (157, 189, 156), mode='nearest').squeeze().numpy()
    img = np.transpose(img, (1, 2, 3, 0))

    orig = nib.load(path_orig_nifti)
    img = nib.Nifti1Image(img, orig.affine, orig.header)
    nib.save(img, save_path)


def train(model, device, train_loader, optimizer):
    model.train()

    train_loss = 0
    for batch_idx, sample in enumerate(train_loader):
        data = sample['input']
        target = sample['target']
        orig_path = sample['location']

        # # target sanity check after
        # save_path = '/share4/bayrakrg/tractEM/postprocessing/deep_tracing/nifti/' + orig_path[0][-17:-11] + '_ttarget.nii'
        # save_nifti(target, orig_path[0], save_path)  # batching therefore we need the 0th

        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        target = target.to(device)

        ############################
        # Different Loss Functions #
        ############################

        # DICE BackgroundLoss
        criterion = DICELoss().to(device)
        loss = criterion(output, target)  # ignore the background for now


        # # DICE BackgroundLoss
        # criterion = DICELoss().to(device)
        # seed_loss = criterion(output[:, 1, :, :, :], target[:, 1, :, :, :])  # ignore the background for now
        # roi_loss = criterion(output[:, 2, :, :, :], target[:, 2, :, :, :])
        # loss = (seed_loss + roi_loss)/2


        # # DICE LOSS for each segmentation ( weighted DICE)
        # criterion = DICELoss().to(device)
        # seed_loss = criterion(output[:, 1, :, :, :], target[:, 1, :, :, :])
        # roi_loss = criterion(output[:, 1, :, :, :], target[:, 2, :, :, :])
        # roa_loss = criterion(output[:, 3, :, :, :], target[:, 3, :, :, :])
        # loss = (4*seed_loss + 4*roi_loss + roa_loss)/9

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # target sanity check after
        # print(output.shape)
        save_path = '/share4/bayrakrg/tractEM/postprocessing/deep_tracing/nifti/' + orig_path[0][-17:-11] + '_toutput.nii'
        save_nifti(output, orig_path[0], save_path)

        break  # second sanity check point

    train_loss /= len(train_loader.dataset) / train_loader.batch_size
    # print('\tTraining set: Average loss: {:.4f}'.format(train_loss), end='')
    return train_loss


def val(model, device, val_loader):
    model.eval()

    val_loss = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            data = sample['input']
            target = sample['target']
            orig_path = sample['location']

            data = data.to(device)
            output = model(data)
            target = target.to(device)

            criterion = DICELoss().to(device)
            vloss = criterion(output[:, 1, :, :, :], target[:, 1, :, :, :])

            val_loss += vloss.item()

            # target sanity check after
            # print(output.shape)
            save_path = '/share4/bayrakrg/tractEM/postprocessing/deep_tracing/nifti/' + orig_path[0][-17:-11] + '_voutput.nii'
            save_nifti(output, orig_path[0], save_path)

            break  # second sanity check point

    val_loss /= len(val_loader.dataset) / val_loader.batch_size
    # print('\tValidation set: Average loss: {:.4f}'.format(val_loss), end='')
    return val_loss


class DICELoss(torch.nn.Module):
    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, input, target):
        smooth = 1.

        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        return 2 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
