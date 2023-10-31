
import argparse
import logging
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
# from utils.data_loading import BasicDataset, CarvanaDataset
from utils.SpineDataSet import SpineDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import SCN


# dir_img = Path("E:/Dataset/Vesta/Landmark/pngs")
# dir_mask = Path("E:/Dataset/Vesta/Landmark/labels")
dir_img = Path('/home/jmed/wmz/DataSet/Spine2D/Landmark/pngs/')
dir_mask = Path('/home/jmed/wmz/DataSet/Spine2D/Landmark/labels/')
dir_checkpoint = Path('./checkpoints/')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1.0):

    # 1. Create dataset
    dataset = SpineDataset(images_dir=dir_img, masks_dir=dir_mask, scale=img_scale)
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=5e-8)
    criterion = nn.MSELoss()
    model_str = ''
    log_saved_dir = os.path.join('runs', model_str) if model_str != '' else None
    writer = SummaryWriter(log_dir=log_saved_dir, comment=f'Class_{net.n_classes}_Epochs_{epochs}')
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {  + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['heatmap']

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                pred, HLA, HSC = net(images)
                loss = criterion(pred, true_masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                writer.add_scalar('Loss/Train_MSE', loss, global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                # Evaluation round
                if global_step % (n_train // (10 * batch_size)) == 0:
                    val_loss = evaluate(net, val_loader, device)

                    logging.info('Validation loss: {}'.format(val_loss))
                    writer.add_scalar('Loss/Valid_MSE', val_loss, global_step)
                    writer.add_scalar('Loss/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('Loss/epoch_loss', epoch_loss, epoch)
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the SCNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = SCN(in_channels=1, num_classes=25, spatial_act="sigmoid")

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)

