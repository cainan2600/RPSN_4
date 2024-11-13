import torch
import os

def checkpoints(model, epoch, optimizer, loss, checkpoint_dir):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir, f'checkpoint-epoch{epoch}.pt')
    torch.save(state, filename)