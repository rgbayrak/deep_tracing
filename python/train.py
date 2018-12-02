import torch
import python.subjectlist as subl
from python.file_parse import parse_file
import os
import argparse
import torchsrc as torchsrc
import random


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


opt = parser.parse_args()
print(opt)

# hyper parameters
epoch_num = opt.epoch
batch_size = 1
lmk_num = 2
learning_rate = opt.lr  #0.0001

data_list = '/home/hansencb/Dropbox/VandyFall2018/deep_medical_imaging/deep_tracing/sublist.txt'
out = './'

train_dict = parse_file(data_list)
keys = list(train_dict.keys())
test_dict = {}

for i in range(5):
	idx = random.randint(0, len(keys)-1)
	test_dict[keys[idx]] = train_dict[keys[idx]]
	del train_dict[keys[idx]]


# load image
train_set = torchsrc.imgloaders.pytorch_loader(train_dict, num_labels=lmk_num)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=1)

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
)


print("==start training==")

start_epoch = 0
start_iteration = 1
trainer.epoch = start_epoch
trainer.iteration = start_iteration
trainer.train_epoch()







