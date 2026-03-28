#
# COPY ADAPTED FROM voxelmorph/scripts/train.py 
# 

# Core library imports
import argparse
from typing import Sequence
from pathlib import Path

# Third-party imports
import numpy as np
import nibabel as nib
import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import neurite as ne
import pandas as pd


import sys

# Local imports
#sys.path.append("/mnt/hd1/code/github/hello-voxelmorph/voxelmorph/torch/voxelmorph")
sys.path.append("/cvibraid/cvib2/apps/personal/pteng/github/hello-voxelmorph/voxelmorph/torch/voxelmorph")

import voxelmorph as vxm
from voxelmorph.py.utils import jacobian_determinant

import torchio as tio

class VxmIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for infinite VoxelMorph registration data.
    """

    def __init__(self, csv_file, device: str = 'cpu') -> None:
        """
        Parameters
        ----------
        device : str
            Device to place tensors on.
        """
        self.device = device
        self.df = pd.read_csv(csv_file)
        self._get_vol_paths()

    def __iter__(self):
        """
        Generate infinite stream of random volume pairs.

        Yields
        ------
        dict
            A dictionary containing the source and target volumes.
        """
        while True:
            idx1, idx2 = np.random.randint(0, len(self.folder_abspaths), size=2)

            # Get paths
            source_path = self.folder_abspaths[idx1]
            target_path = self.folder_abspaths[idx2]


            transform = tio.Resize((128,128,128))
            rescale = tio.RescaleIntensity(out_min_max=(-1,1),in_min_max=(-1000,1000))

            source_nii = tio.ScalarImage(source_path)
            source_nii = transform(rescale(source_nii))

            target_nii = tio.ScalarImage(source_path)
            target_nii = transform(rescale(target_nii))

            source = source_nii.tensor
            target = target_nii.tensor

            yield {'source': source, 'target': target}

            # # Get niftis
            # source_nii = nib.load(source_path)
            # target_nii = nib.load(target_path)

            # source = torch.from_numpy(source_nii.get_fdata()).float().unsqueeze(0)
            # target = torch.from_numpy(target_nii.get_fdata()).float().unsqueeze(0)

            #yield {'source': source, 'target': target}

    def _get_vol_paths(self) -> None:
        """
        Get the absolute paths of the volume folders.
        """
        self.folder_abspaths = self.df.NiftiFile.to_list()


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_loss_fn: nn.Module,
    grad_loss_fn: nn.Module,
    loss_weights: Sequence[float],
    steps_per_epoch: int,
    device: str = 'cuda'
) -> float:
    """
    Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The VoxelMorph model to train.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use for training.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    image_loss_fn : nn.Module
        The image loss function to use.
    grad_loss_fn : nn.Module
        The gradient loss function to use.
    loss_weights : Sequence[float]
        The weights for the image and gradient losses.
    steps_per_epoch : int
    """

    model.train()
    total_loss = 0.0

    for _ in range(steps_per_epoch):
        batch = next(dataloader)
        optimizer.zero_grad()

        # Move to device in training loop (not dataloader/dataset!)
        source = batch['source'].to(device)
        target = batch['target'].to(device)

        # Get the displacement and the warped source image from the model
        displacement, warped_source = model(
            source,
            target,
            return_warped_source=True,
            return_field_type='displacement'
        )

        disp = np.moveaxis(displacement.cpu().detach().numpy().squeeze(),[0,1,2,3],[3,0,1,2])
        print(disp.shape)
        # jdet need to compute per case?
        #jdet = jacobian_determinant(disp)

        img_loss = image_loss_fn(target, warped_source)
        grad_loss = grad_loss_fn(displacement)

        loss = loss_weights[0] * img_loss + loss_weights[1] * grad_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / steps_per_epoch


def main():
    parser = argparse.ArgumentParser(description='Train 3D VoxelMorph on OASIS data')
    parser.add_argument('csv_file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100_000, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda', type=float, dest='lambda_param', default=0.01)
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--save-every', type=int, default=10, help='Checkpoint every N epochs')
    args = parser.parse_args()

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Create model
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=[16, 16, 16, 16, 16],
        integration_steps=0,
    ).to(device)

    # Setup losses and optimizer
    image_loss_fn = ne.nn.modules.MSE()
    grad_loss_fn = ne.nn.modules.SpatialGradient('l2')
    loss_weights = [1.0, args.lambda_param]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create dataloader
    train_dataset = VxmIterableDataset(args.csv_file,device=device)
    train_loader = iter(
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f'Training for {args.epochs} epochs...')
    best_loss = float('inf')
    for epoch in tqdm(range(args.epochs), desc='Epochs'):

        # Train for one epoch
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            image_loss_fn=image_loss_fn,
            grad_loss_fn=grad_loss_fn,
            loss_weights=loss_weights,
            steps_per_epoch=args.steps_per_epoch,
            device=device
        )

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.6f}')

        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:

            checkpoint_path = output_dir / f'checkpoint_epoch{epoch + 1}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / 'best.pt'
            torch.save(model.state_dict(), best_path)

    # Save final model
    final_path = output_dir / 'final.pt'
    torch.save(model.state_dict(), final_path)
    print(f'Final model saved to {final_path}')


if __name__ == '__main__':
    main()
