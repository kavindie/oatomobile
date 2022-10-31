# Copyright 2020 The OATomobile Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trains the deep imitative model on expert demostrations."""

import os
import argparse
from contextlib import contextmanager
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from flomo.utils_CARLA.datasetCARLA_flomo_new import MultiMyDataset

from typing import Mapping

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from absl import app
from absl import flags
from absl import logging

from oatomobile.baselines.torch.dim.model import ImitativeModel
from oatomobile.datasets.carla import CARLADataset
from oatomobile.torch import types
from oatomobile.torch.loggers import TensorBoardLogger
from oatomobile.torch.savers import Checkpointer

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS


class Config:
    # training
    lr = 0.001
    epochs = 150
    batch_size = 128
    val_split = 0.1
    sH = 480
    sW = 640

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    patience = 5

    condition = None

    # flag for utility functions
    rel_coords = True
    noise = True
    norm_rotation = True
    global_rotation = False

    # model values
    alpha = 0.6
    beta = 0.02
    gamma = 0.002

    # testing
    num_plots = 100
    num_samples = 20

    # dataset
    dataset_dir = '../udmt/CARLA_Data/OutputData'
    csv_file = 'newdata_unique_capped.csv'
    sequence_length = 16  # original 20
    min_sequence_length = 10
    observed_history = 8
    pred_steps = sequence_length - observed_history


def parse_commandline():
    parser = argparse.ArgumentParser(description='Run training motion prediction network.')
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/kavindie/Documents/Research_UTS/Python_in_ML/udmt/CARLA_Data/OutputData/Second/newdata_unique_capped.csv',
                        help='Directory to the dataset')
    parser.add_argument('--output_dir', type=str, default='./oatomobile/baselines/torch/dim/test2',
                        help='The full path to the output directory (for logs, ckpts).')
    parser.add_argument('--batch_size', type=int, default=128,
                        help="The batch size used for training the neural network.")
    parser.add_argument('--num_epochs', type=int, default=150,
                        help="The number of training epochs for the neural network.")
    parser.add_argument('--save_model_frequency', type=int, default=20,
                        help="The number epochs between saves of the model.")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="The ADAM learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="The L2 penalty (regularization) coefficient.")
    parser.add_argument('--clip_gradients', default=False, help="If True it clips the gradients norm to 1.0.")
    parser.add_argument('--num_timesteps_to_keep', type=int, default=8,
                        help="The numbers of time-steps to keep from the target, with downsampling.")
    parser.add_argument('--load', type=str, default=None, help="Are you loading a model?")
    args = parser.parse_args()
    Config.batch_size = args.batch_size
    Config.epochs = args.num_epochs
    return args


def testOrientation(batch):
    import transforms3d
    import numpy as np
    import matplotlib as mpl
    mpl.use('tkagg')
    traj_in = batch['traj_in_rotated_glob']  # batch['traj_in_rotated']
    traj_in = traj_in[batch['mask'].bool()]
    traj_out = batch['traj_out_rotated_glob']  # batch['traj_in_rotated']
    traj_out = traj_out[batch['mask'].bool()]
    plt.plot(batch['traj_in'][0, :, 0], batch['traj_in'][0, :, 1], '.-r', alpha=0.5)
    plt.plot(batch['traj_out'][0, :, 0], batch['traj_out'][0, :, 1], '.-g', alpha=0.5)
    plt.axis('equal')
    plt.show()
    plt.plot(traj_in[0, :, 0], traj_in[0, :, 1], '.-r', alpha=0.8)
    plt.plot(traj_out[0, :, 0], traj_out[0, :, 1], '.-g', alpha=0.8)
    plt.show()
    full_traj = torch.cat((traj_in, traj_out), dim=-2)
    full_traj = full_traj - traj_in[0, -1]
    plt.plot(full_traj[0, :, 0], full_traj[0, :, 1], '.-c', alpha=1.0)
    plt.show()
    current_location = torch.cat((traj_in[0, -1, :], torch.zeros([1])))
    full_traj = torch.cat((batch['traj_in'], batch['traj_out']), dim=-2)
    world_locations = full_traj[0]
    world_locations = torch.cat((torch.as_tensor(world_locations), torch.zeros(16, 1)), dim=-1)
    world_locations = np.atleast_2d(world_locations)
    R = transforms3d.euler.euler2mat(0, 0, batch['global_orientation']).T
    local_locations = np.dot(a=R, b=(world_locations - current_location.numpy()).T).T
    plt.plot(local_locations[:, 0], local_locations[:, 1], '.-y')
    plt.show()


def main():
    args = parse_commandline()
    logging.debug(args)
    # Parses command line arguments.
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    save_model_frequency = args.save_model_frequency
    num_timesteps_to_keep = args.num_timesteps_to_keep
    weight_decay = args.weight_decay
    clip_gradients = args.clip_gradients
    noise_level = 1e-2

    # Determines device, accelerator.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # pylint: disable=no-member

    # Creates the necessary output directory.
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Initializes the model and its optimizer.
    output_shape = [num_timesteps_to_keep, 2]
    model = ImitativeModel(output_shape=output_shape).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    writer = TensorBoardLogger(log_dir=log_dir)
    checkpointer = Checkpointer(model=model, ckpt_dir=ckpt_dir)

    def transform(batch: Mapping[str, types.Array]) -> Mapping[str, torch.Tensor]:
        """Preprocesses a batch for the model.

    Args:
      batch: (keyword arguments) The raw batch variables.

    Returns:
      The processed batch.
    """
        # Sends tensors to `device`.
        batch = {key: tensor.to(device) for (key, tensor) in batch.items()}
        # Preprocesses batch for the model.
        batch = model.transform(batch)
        return batch

    # My additions
    data_train = ["Third", "Sixth", "Fourth", "Fifth"]
    data_val = ["First"]
    data_test = ["Second"]

    dataloader_train = DataLoader(
        MultiMyDataset(data_train, Config, operation='train', FloMo_train=False),
        batch_size=Config.batch_size, shuffle=True, num_workers=0, generator=torch.default_generator
    )

    dataloader_val = DataLoader(
        MultiMyDataset(data_val, Config, operation='validate', FloMo_train=False),
        batch_size=Config.batch_size, shuffle=False, num_workers=0, generator=torch.default_generator
    )

    dataloader_test = DataLoader(
        MultiMyDataset(data_test, Config, operation='test', FloMo_train=False),
        batch_size=1, shuffle=False, num_workers=0, generator=torch.default_generator
    )

    # Theoretical limit of NLL.
    nll_limit = -torch.sum(  # pylint: disable=no-member
        D.MultivariateNormal(
            loc=torch.zeros(output_shape[-2] * output_shape[-1]),  # pylint: disable=no-member
            scale_tril=torch.eye(output_shape[-2] * output_shape[-1]) *  # pylint: disable=no-member
                       noise_level,  # pylint: disable=no-member
        ).log_prob(torch.zeros(output_shape[-2] * output_shape[-1])))  # pylint: disable=no-member

    def train_step(
            model: ImitativeModel,
            optimizer: optim.Optimizer,
            batch: Mapping[str, torch.Tensor],
            clip: bool = False,
    ) -> torch.Tensor:
        """Performs a single gradient-descent optimisation step."""
        # Resets optimizer's gradients.
        optimizer.zero_grad()

        # Adding my stuff
        batch_size = Config.batch_size
        # LET'S DO SOME PREPROCESSING
        # testOrientation(batch)
        visual_features = batch['image_resNet+grid'][[batch['mask'].bool()]]
        traj_in = batch['traj_in_rotated_glob'][batch['mask'].bool()]  # batch['traj_in_rotated']
        traj_out = batch['traj_out_rotated_glob'][batch['mask'].bool()]  # batch['traj_in_rotated']
        full_traj = torch.cat((traj_in, traj_out), dim=-2)
        full_traj = full_traj - traj_in[0, -1]
        traj_in = full_traj[:, :8, :]
        traj_out = full_traj[:, 8:, :]
        traj_in = traj_in[:, 1:] - traj_in[:, :-1]

        # Perturb target.
        y = torch.normal(  # pylint: disable=no-member
            mean=traj_out[..., :2],
            std=torch.ones_like(traj_out[..., :2]) * noise_level,  # pylint: disable=no-member
        )

        # Forward pass from the model.
        z = model._params(
            velocity=traj_in,
            visual_features=visual_features,
            traffic_light_state=batch['action_gt'][[batch['mask'].bool()]],
        )

        _, log_prob, logabsdet = model._decoder._inverse(y=y, z=z)

        # Calculates loss (NLL).
        loss = -torch.mean(log_prob - logabsdet, dim=0)  # pylint: disable=no-member

        # Backward pass.
        loss.backward()

        # Clips gradients norm.
        if clip:
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

        # Performs a gradient descent step.
        optimizer.step()

        return loss

    def train_epoch(
            model: ImitativeModel,
            optimizer: optim.Optimizer,
            dataloader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Performs an epoch of gradient descent optimization on `dataloader`."""
        model.train()
        loss = 0.0
        # tt = 0
        with tqdm.tqdm(dataloader) as pbar:
            for batch in pbar:
                # Prepares the batch.
                batch = transform(batch)
                # Performs a gradien-descent step.
                loss += train_step(model, optimizer, batch, clip=clip_gradients)
                # tt += 1
                # if tt == 20:
                #     return loss/20
        return loss / len(dataloader)

    def evaluate_step(
            model: ImitativeModel,
            batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluates `model` on a `batch`."""
        # Forward pass from the model.
        # LET'S DO SOME PREPROCESSING
        # testOrientation(batch)
        visual_features = batch['image_resNet+grid'][[batch['mask'].bool()]]
        traj_in = batch['traj_in_rotated_glob'][batch['mask'].bool()]  # batch['traj_in_rotated']
        traj_out = batch['traj_out_rotated_glob'][batch['mask'].bool()]  # batch['traj_in_rotated']
        full_traj = torch.cat((traj_in, traj_out), dim=-2)
        full_traj = full_traj - traj_in[0, -1]
        traj_in = full_traj[:, :8, :]
        traj_out = full_traj[:, 8:, :]
        traj_in = traj_in[:, 1:] - traj_in[:, :-1]

        z = model._params(
            velocity=traj_in,
            visual_features=visual_features,
            traffic_light_state=batch['action_gt'][[batch['mask'].bool()]],
        )
        _, log_prob, logabsdet = model._decoder._inverse(
            y=traj_out[..., :2],
            z=z,
        )

        # Calculates loss (NLL).
        loss = -torch.mean(log_prob - logabsdet, dim=0)  # pylint: disable=no-member

        return loss

    def evaluate_epoch(
            model: ImitativeModel,
            dataloader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Performs an evaluation of the `model` on the `dataloader."""
        model.eval()
        loss = 0.0
        # tt = 0
        with tqdm.tqdm(dataloader) as pbar:
            for batch in pbar:
                # Prepares the batch.
                batch = transform(batch)
                # Accumulates loss in dataset.
                with torch.no_grad():
                    loss += evaluate_step(model, batch)
                # tt += 1
                # if tt == 20:
                #     return loss / 20
        return loss / len(dataloader)

    def write(
            model: ImitativeModel,
            dataloader: torch.utils.data.DataLoader,
            writer: TensorBoardLogger,
            split: str,
            loss: torch.Tensor,
            epoch: int,
    ) -> None:
        """Visualises model performance on `TensorBoard`."""
        # Gets a sample from the dataset.
        for batch in dataloader:
            # Prepares the batch.
            batch = transform(batch)
            # Turns off gradients for model parameters.
            for params in model.parameters():
                params.requires_grad = False
            # Generates predictions.
            goal = batch['visible_last_point_xy_traverse_rot_globally'][batch['mask'].bool()]
            goal = goal[:, None, :]
            predictions = model(num_steps=20, goal=goal, **batch)
            # Turns on gradients for model parameters.
            for params in model.parameters():
                params.requires_grad = True
            # Logs on `TensorBoard`.
            vf = batch['image_resNet+grid'][batch['mask'].bool()]
            traj_in = batch['traj_in_rotated_glob'][batch['mask'].bool()]
            traj_out = batch['traj_out_rotated_glob'][batch['mask'].bool()]  # batch['traj_in_rotated']
            full_traj = torch.cat((traj_in, traj_out), dim=-2)
            full_traj = full_traj - traj_in[0, -1]
            traj_in = full_traj[:, :8, :]
            gt = full_traj[:, 8:, :]

            writer.log(
                split=split,
                loss=loss.detach().cpu().numpy().item(),
                overhead_features=vf.detach().cpu().numpy()[:8],
                predictions=predictions.detach().cpu().numpy()[:8],
                ground_truth=gt.detach().cpu().numpy()[:8],
                input_traj=traj_in.detach().cpu().numpy()[:8],
                global_step=epoch,
            )
            if split != 'test':
                break

    if args.load is not None:
        if torch.cuda.is_available():
            saved_info = torch.load(args.load)
        else:
            saved_info = torch.load(args.load, map_location=torch.device('cpu'))
        model.load_state_dict(saved_info)
        loss_test = evaluate_epoch(model, dataloader_test)
        write(model, dataloader_test, writer, "test", loss_test, 151)
    else:
        with tqdm.tqdm(range(num_epochs)) as pbar_epoch:
            loss_best = torch.inf
            for epoch in pbar_epoch:
                # Trains model on whole training dataset, and writes on `TensorBoard`.
                loss_train = train_epoch(model, optimizer, dataloader_train)
                write(model, dataloader_train, writer, "train", loss_train, epoch)

                # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
                loss_val = evaluate_epoch(model, dataloader_val)
                write(model, dataloader_val, writer, "val", loss_val, epoch)

                if loss_val < loss_best:
                    loss_best = loss_val
                    checkpointer.save(epoch)

                # Checkpoints model weights.
                if epoch % save_model_frequency == 0:
                    checkpointer.save(epoch)

                # Updates progress bar description.
                pbar_epoch.set_description(
                    "TL: {:.2f} | VL: {:.2f} | THEORYMIN: {:.2f}".format(
                        loss_train.detach().cpu().numpy().item(),
                        loss_val.detach().cpu().numpy().item(),
                        nll_limit,
                    ))


@contextmanager
def cuda_context(cuda=None):
    if cuda is None:
        cuda = torch.cuda.is_available()
    old_tensor_type = torch.cuda.FloatTensor if torch.tensor(0).is_cuda else torch.FloatTensor
    old_generator = torch.default_generator
    torch.set_default_tensor_type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
    torch.default_generator = torch.Generator('cuda' if cuda else 'cpu')
    yield
    torch.set_default_tensor_type(old_tensor_type)
    torch.default_generator = old_generator


if __name__ == "__main__":
    with cuda_context():
        main()
