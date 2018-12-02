import torch
import subjectlist as subl
import os
import argparse
import deep_tracing.torchsrc as torchsrc


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for, default=10')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--finetune',type=bool,default=False,help='fine tuning using true')
parser.add_argument('--fineepoch', type=int, default=5, help='fine tuning starting epoch')


opt = parser.parse_args()
print(opt)

# hyper parameters
epoch_num = opt.epoch
batch_size = 1
lmk_num = 2
learning_rate = opt.lr  #0.0001

finetune = opt.finetune
fineepoch = opt.fineepoch


# define paths
train_list_file = '/home-local/bayrakrg/MDL/deep_tracing/sublist.txt'
val_list_file = '/home-local/bayrakrg/MDL/deep_tracing/sublist.txt'
working_dir = os.path.join('/home-local/bayrakrg/MDL/Ass3_Segmentation/v3D/SLANTbrainSeg-master/Results')
test_img_dir = '/home-local/bayrakrg/MDL/Ass3_Segmentation/assignment3/Testing/img'
finetune_img_dir = '/home-local/bayrakrg/MDL/Ass3_Segmentation/assignment3/Training/img'
finetune_seg_dir = '/home-local/bayrakrg/MDL/Ass3_Segmentation/assignment3/Training/label'


# make img list

if finetune == True:
	out = os.path.join(working_dir, 'finetune_out')
	mkdir(out)
	train_img_files = subl.get_sub_list(finetune_img_dir)
	train_seg_files = subl.get_sub_list(finetune_seg_dir)
	train_dict = {}
	train_dict['img_files'] = train_img_files
	train_dict['seg_files'] = train_seg_files

else:
	out = os.path.join(working_dir, 'latest')
	mkdir(out)
	train_img_files, train_seg_files = subl.get_sub_from_txt(train_list_file)
	train_dict = {}
	train_dict['img_files'] = train_img_files
	train_dict['seg_files'] = train_seg_files

	val_img, val_seg = subl.get_sub_from_txt(val_list_file)
	val_dict = {}
	val_dict['img_files'] = val_img
	val_dict['seg_files'] = val_seg


test_img = subl.get_sub_list(test_img_dir)
test_dict = {}
test_dict['img_files'] = test_img



# load image
train_set = torchsrc.imgloaders.pytorch_loader(train_dict, num_labels=lmk_num)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=1)
# val_set = torchsrc.imgloaders.pytorch_loader(val_dict, num_labels=lmk_num)
# val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,shuffle=False,num_workers=1)
test_set = torchsrc.imgloaders.pytorch_loader(test_dict,num_labels=lmk_num)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=1)

# load network
model = torchsrc.models.UNet3D(in_channel=1, n_classes=lmk_num)
# model = torchsrc.models.VNet()

# print_network(model)
#
# load optimizer
optim = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999))
# optim = torch.optim.SGD(model.parameters(), lr=learning_curve() _rate, momentum=0.9)

# load CUDA
cuda = torch.cuda.is_available()
torch.manual_seed(1)
if cuda:
	torch.cuda.manual_seed(1)
	model = model.cuda()

# load trainer
trainer = torchsrc.Trainer(
	cuda=cuda,
	model=model,
	optimizer=optim,
	train_loader=train_loader,
	# val_loader=val_loader,
	test_loader=test_loader,
	out=out,
	max_epoch = epoch_num,
	batch_size = batch_size,
	lmk_num = lmk_num,
	finetune = finetune,
	fineepoch = fineepoch
)


print("==start training==")

start_epoch = 0
start_iteration = 1
trainer.epoch = start_epoch
trainer.iteration = start_iteration
trainer.train_epoch()







