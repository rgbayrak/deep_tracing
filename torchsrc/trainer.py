import datetime
import os
import os.path as osp
import pandas as pd
import numpy as np
import pytz
import scipy.misc
import nibabel as nib
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt

def saveOneImg(img,path,cate_name,sub_name,surfix,):
    filename = "%s-x-%s-x-%s.png"%(cate_name,sub_name,surfix)
    file = os.path.join(path,filename)
    scipy.misc.imsave(file, img)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)


def weighted_center(input,threshold=0.75):
    # m= torch.nn.Tanh()
    # input = m(input)

    input = torch.add(input, -input.min().expand(input.size())) / torch.add(input.max().expand(input.size()), -input.min().expand(input.size()))
    m = torch.nn.Threshold(threshold, 0)
    input = m(input)
    grid = np.meshgrid(range(input.size()[0]), range(input.size()[1]), indexing='ij')
    x0 = torch.mul(input, Variable(torch.from_numpy(grid[1]).float().cuda())).sum() / input.sum()
    y0 = torch.mul(input, Variable(torch.from_numpy(grid[0]).float().cuda())).sum() / input.sum()
    return x0, y0


def save_images(results_epoch_dir,data,sub_name,cate_name,pred_lmk,target=None):
    saveOneImg(data[0, 0, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_trueGray")
    for i in range(pred_lmk.size()[1]):
        saveOneImg(pred_lmk[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_pred%d" % (i))
        if not (target is None):
            saveOneImg(target[0, i, :, :].data.cpu().numpy(), results_epoch_dir, cate_name,sub_name, "_true%d" % (i))


def dice_loss_3d(input, target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, should have same size as the input
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
    num = torch.sum(num, dim=2)  #
    num = torch.sum(num, dim=0)# b,c

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)
    den1 = torch.sum(den1, dim=2)  # b,c,1,1
    den1 = torch.sum(den1, dim=0)

    den2 = target * target  # --g^2
    den2 = torch.sum(den2, dim=3)
    den2 = torch.sum(den2, dim=2)  # b,c,1,1
    den2 = torch.sum(den2, dim=0)

    dice = 2 * (num / (den1 + den2 + 0.0000001))
    dice_eso = dice[0:]  # we ignore background dice val, and take the foreground
    dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    dice_total = dice_total
    return 1 + dice_total


def dice(pred, target):
    pred = np.argmax(pred, axis=1)
    target = np.argmax(target, axis=1)
    pred = pred.flatten()
    target = target.flatten()

    denom = (np.sum(pred) + np.sum(target))

    if denom != 0:
        return np.sum(pred[target==1])*2.0 / (np.sum(pred) + np.sum(target))
    return 1


class Trainer(object):

    def __init__(self, cuda, model, optimizer=None,
                train_loader=None, val_loader=None, test_loader=None,lmk_num=None,
                train_root_dir=None,out=None, max_epoch=None, batch_size=None,
                size_average=False, interval_validate=None,	fineepoch=None,
	            finetune=False, compete = False, onlyEval=False):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        self.interval_validate = interval_validate
        self.size_average = size_average

        self.train_root_dir = train_root_dir
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.lmk_num = lmk_num


        self.max_epoch = max_epoch
        self.epoch = 0
        self.iteration = 0
        self.best_mean_iu = 0
        self.batch_size = batch_size

        self.finetune = finetune
        self.fineepoch = fineepoch

    def validate(self):
        self.model.train()
        out = osp.join(self.out, 'seg_output')
        out_vis = osp.join(self.out, 'visualization')
        results_epoch_dir = osp.join(out, 'epoch_%04d' % self.epoch)
        mkdir(results_epoch_dir)

        for batch_idx, (data,target,sub_name) in tqdm.tqdm(
                # enumerate(self.test_loader), total=len(self.test_loader),
                enumerate(self.test_loader), total=len(self.test_loader),
                desc='Valid epoch=%d' % self.epoch, ncols=80,
                leave=False):

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target,volatile=True)

            pred = self.model(data)
            lbl_pred = pred.data.max(1)[1].cpu().numpy()[:,:, :].astype('uint8')
            batch_num = lbl_pred.shape[0]
            for si in range(batch_num):
                curr_sub_name = sub_name[si]
                out_img_dir = os.path.join(results_epoch_dir, 'seg')
                mkdir(out_img_dir)
                out_nii_file = os.path.join(out_img_dir,('%s_seg.nii.gz'%(curr_sub_name)))
                seg_img = nib.Nifti1Image(lbl_pred[si], affine=np.eye(4))
                nib.save(seg_img, out_nii_file)



    def train(self):
        self.model.train()

        out = osp.join(self.out, 'visualization')
        mkdir(out)
        log_file = osp.join(out, 'training_loss.txt')
        fv = open(log_file, 'a')
        loss_list = []
        dice_list = []
        for batch_idx, (data, target, sub_name) in tqdm.tqdm(
            enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            if self.cuda:
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)

            pred = self.model(data)
            self.optim.zero_grad()

            loss = dice_loss_3d(pred, target)
            fv.write('epoch=%d,loss=%.4f \n'%(self.epoch, loss.item()))
            loss.backward()
            self.optim.step()

            pred_copy = torch.Tensor.cpu(pred)
            target = torch.Tensor.cpu(target)
            pred_copy = pred_copy.detach().numpy()
            target = target.detach().numpy()
            dice_score = dice(pred_copy, target)

            dice_list = np.append(dice_list, dice_score)
            loss_list.append(loss.data.cpu().detach().numpy().tolist())
        print('\nTrain Loss: {:.4f}   Avg. Dice: {:.4f}    Median Dice: {:.4f}    Stddev Dice: {:.4f}\n'.format(np.mean(loss_list), np.mean(dice_list),
                                                                                                              np.median(dice_list),
                                                                                                              np.std(dice_list)))
        fv.close()


    def val(self):
        self.model.train()

        out = osp.join(self.out, 'visualization')
        mkdir(out)
        log_file = osp.join(out, 'validation_loss.txt')
        fv = open(log_file, 'a')

        loss_vlist = []
        dice_vlist = []
        for batch_idx, (data, target, sub_name) in tqdm.tqdm(
            enumerate(self.val_loader), total=len(self.val_loader),
                desc='Val epoch=%d' % self.epoch, ncols=80, leave=False):

            if self.cuda:
                vdata, vtarget = data.cuda(), target.cuda()

            vdata, vtarget = Variable(vdata), Variable(vtarget)

            vpred = self.model(vdata)
            self.optim.zero_grad()

            vloss = dice_loss_3d(vpred, vtarget)
            fv.write('epoch=%d,loss=%.4f \n'%(self.epoch, vloss.item()))

            pred_copy = torch.Tensor.cpu(vpred)
            target = torch.Tensor.cpu(vtarget)
            pred_copy = pred_copy.detach().numpy()
            target = target.detach().numpy()
            vdice_score = dice(pred_copy, target)

            dice_vlist = np.append(dice_vlist, vdice_score)
            loss_vlist.append(vloss.data.cpu().detach().numpy().tolist())
        print('Validation Loss: {:.4f}   Avg. Dice: {:.4f}    Median Dice: {:.4f}    Stddev Dice: {:.4f}\n'.format(np.mean(loss_vlist), np.mean(dice_vlist),
                                                                                                                   np.median(dice_vlist),
                                                                                                                   np.std(dice_vlist)))
        fv.close()


    def train_epoch(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            out = osp.join(self.out, 'models')
            mkdir(out)

            model_pth = '%s/model_epoch_%04d.pth' % (out, epoch)

            if self.finetune:
                old_out = out.replace('finetune_out','test_out4')
                old_model_pth = '%s/model_epoch_%04d.pth' % (old_out, self.fineepoch)
                self.model.load_state_dict(torch.load(old_model_pth))


            if os.path.exists(model_pth):
                print("start load")
                self.model.load_state_dict(torch.load(model_pth))\
                #print("finsih load")
                # self.validate()
            else:
                self.train()
                self.val()
                if epoch >= 20:
                    self.validate()

                torch.save(self.model.state_dict(), model_pth)

                # torch.save(self.model.state_dict(), model_pth)

    def val_epoch(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Val', ncols=80):
            self.epoch = epoch
            train_root_dir = osp.join(self.train_root_dir, 'models')

            model_pth = '%s/model_epoch_%04d.pth' % (train_root_dir, epoch)
            if os.path.exists(model_pth):
                self.model.load_state_dict(torch.load(model_pth))
                self.val()


    def test_epoch(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Test', ncols=80):
            self.epoch = epoch
            train_root_dir = osp.join(self.train_root_dir, 'models')

            model_pth = '%s/model_epoch_%04d.pth' % (train_root_dir, epoch)
            if os.path.exists(model_pth):
                self.model.load_state_dict(torch.load(model_pth))
                self.validate()


