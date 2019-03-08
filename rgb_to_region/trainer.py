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
    val_loss_file = '/share4/bayrakrg/tractEM/postprocessing/deep_tracing/val_loss.txt'

    # new model
    if int(sys.argv[1]) == -1:
        f = open(train_loss_file, 'w')
        f.close()
        g = open(val_loss_file, 'w')
        g.close()

    train_dataset = dataset(['/share4/bayrakrg/tractEM/postprocessing/rgb_files/HCP', '/share4/bayrakrg/tractEM/postprocessing/rgb_files/BLSA'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    val_dataset = dataset(['/share4/bayrakrg/tractEM/postprocessing/rgb_files/BLSA19'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)

    model_file = '/share4/bayrakrg/tractEM/postprocessing/deep_tracing/' + 'epoch_{}'

    in_channel = 3
    n_classes = 3
    model = UNet3D(in_channel, n_classes).to(device)
    lr = 0.000001 #reduce the learning rate
    num_epochs = 100
    total_step = len(train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    start_epoch = 1

    # fine tuning >> change system argv to the episode you want to start with
    if int(sys.argv[1]) != -1:
        model.load_state_dict(torch.load(model_file.format(sys.argv[1])))
        start_epoch = int(sys.argv[1]) + 1

    for epoch in range(start_epoch, num_epochs):
        print('\nEpoch {}: '.format(epoch))

        # # decaying learning rate
        # if epoch % 10 == 0:
        #     lr = lr/3

        # Training
        train_loss = train(model, device, train_loader, optimizer)

        # save train loss file
        with open(train_loss_file, "a") as file:
            file.write(str(train_loss))
            file.write('\n')

        # save model
        if epoch % 1 == 0:
            with open(model_file.format(epoch), 'wb') as f:
                torch.save(model.state_dict(), f)

        if epoch % 1 == 0:
            # Validating
            val_loss = val(model, device, val_loader)

            # save validation loss file
            with open(val_loss_file, "a") as file:
                file.write(str(val_loss))
                file.write('\n')

            print("Epoch [{}/{}], Total Step [{}] ---> Loss: {:.4f}, Val_Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, total_step, train_loss, val_loss))

            # scheduler.step(val_loss.cpu().data.numpy())

        break  # break for the sanity check point

if __name__ == '__main__':
    main()