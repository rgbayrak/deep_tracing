from __future__ import print_function, division
import os
import torch

import numpy as np

import matplotlib.pyplot as plt
import torch.nn.functional as F
import nibabel as nib
import warnings
warnings.filterwarnings("ignore")

def save_nifti(img, path_orig_nifti, save_path):
    img = img.detach().cpu().numpy()
    img = np.squeeze(img)
    img = np.transpose(img, (1, 2, 3, 0))

    # print(np.sum(img[:,:,:,0]))
    # print(np.sum(img[:, :, :, 1]))
    # print(np.sum(img[:, :, :, 2]))

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
        # save_path = '/share4/bayrakrg/tractEM/postprocessing/deep_tracing/nifti/' + orig_path[0][-17:-11] + '_target.nii'
        # save_nifti(target, orig_path[0], save_path)  # batching therefore we need the 0th

        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        target = target.to(device)

        # # Dice Loss
        # criterion = DICELoss().to(device)
        # loss = criterion(output, target)

        # Cross Entropy Loss
        criterion = torch.nn.CrossEntropyLoss().to(device)
        loss = criterion(output, torch.argmax(target, dim=1))

        # # Dice Loss for each segmentation
        # criterion = DICELoss().to(device)
        # seed_loss = criterion(output[:, 0, :, :, :], target[:, 0, :, :, :])
        # roi_loss = criterion(output[:, 1, :, :, :], target[:, 1, :, :, :])
        # roa_loss = criterion(output[:, 2, :, :, :], target[:, 2, :, :, :])
        # loss = (4*seed_loss + 4*roi_loss + roa_loss)/9

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # # target sanity check after
        # save_path = '/share4/bayrakrg/tractEM/postprocessing/deep_tracing/nifti/' + orig_path[0][-17:-11] + '_output.nii'
        # save_nifti(output, orig_path[0], save_path)
        #
        # break  # second sanity check point

    train_loss /= len(train_loader.dataset) / train_loader.batch_size
    print('\tTraining set: Average loss: {:.4f}'.format(train_loss), end='')
    return train_loss


# def val(model, device, train_loader, optimizer):
#     model.eval()
#
#     val_loss = 0
#     for batch_idx, sample in enumerate(train_loader):
#         data = sample['input']
#         target = sample['target']
#
#         data = data.to(device)
#         optimizer.zero_grad()
#
#         output = model(data)
#
#         target = target.to(device)
#         criterion = torch.nn.CrossEntropyLoss().to(device)
#         loss = criterion(output, target)
#
#         val_loss += loss.item()
#
#     val_loss /= len(train_loader.dataset) / train_loader.batch_size
#     print('\tValidation set: Average loss: {:.4f}'.format(val_loss), end='')
#     return val_loss


class DICELoss(torch.nn.Module):
    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, target, input):
        smooth = 1.

        iflat = input.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        return 2 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
