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
        loss = dice_loss_3d(output, target)

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


def val(model, device, train_loader, optimizer):
    model.eval()

    val_loss = 0
    for batch_idx, sample in enumerate(train_loader):
        data = sample['input']
        target = sample['target']

        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)

        target = target.to(device)
        loss = dice_loss_3d(output, target)

        val_loss += loss.item()

    val_loss /= len(train_loader.dataset) / train_loader.batch_size
    print('\tValidation set: Average loss: {:.4f}'.format(val_loss), end='')
    return val_loss


def dice_loss_3d(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxWxD representing probabilities for each class
    target is a 1-hot representation of the ground truth, should have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 5, "Input must be a 5D Tensor."
    # uniques = np.unique(target.numpy())
    # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"
    target = target.view(target.size(0), target.size(1), target.size(2), -1)
    input = input.view(input.size(0), input.size(1), input.size(2), -1)
    probs = F.softmax(input, dim=1)

    num = probs * target  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=0)  # b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=0)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=0)

    dice = 2 * (num / (den1 + den2 + 0.0000001))
    dice_eso = dice[0:]  # we ignore background dice val, and take the foreground
    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

    return 1 + dice_total
