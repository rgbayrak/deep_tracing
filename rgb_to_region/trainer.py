from data_loader import *
from models import *
from utils import *
import sys
import torch
from torch.utils.data import Dataset


def main():
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loss_file = '/share4/bayrakrg/tractEM/postprocessing/deep_tracing/train_loss.txt'

    # new model or pre-trained
    if int(sys.argv[1]) == -1:
        f = open(train_loss_file, 'w')
        f.close()

    train_dataset = dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, **kwargs)

    model_file = '/share4/bayrakrg/tractEM/postprocessing/deep_tracing/' + 'epoch_{}'

    in_channel = 3
    n_classes = 3
    model = UNet3D(in_channel, n_classes).to(device)
    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    if int(sys.argv[1]) != -1:
        model.load_state_dict(torch.load(model_file.format(sys.argv[1])))
        start_epoch = int(sys.argv[1]) + 1

    for epoch in range(start_epoch, 100):
        print('\nEpoch {}: '.format(epoch))

        train_loss = train(model, device, train_loader, optimizer)

        with open(train_loss_file, "a") as file:
            file.write(str(train_loss))
            file.write('\n')

        if epoch % 1 == 0:
            with open(model_file.format(epoch), 'wb') as f:
                torch.save(model.state_dict(), f)

        # break # sanity check point

if __name__ == '__main__':
    main()
