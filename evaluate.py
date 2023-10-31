import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    avg_loss = 0
    criterion = nn.MSELoss()

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['heatmap']  # batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            pred, _, _ = net(image)

            # convert to one-hot format
            loss = criterion(pred, mask_true)
            avg_loss += loss

           

    net.train()
    return avg_loss / num_val_batches
