import os
import logging
from tqdm import tqdm
from PIL import Image
from typing import Callable, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from eval import evaluate
from augmentation.augmentations import get_weak_augmentation, get_normalizer
from datasets.config import IMG_SIZE
from utils.train import EMA, apply_wd, linear_rampup, set_bn_running_updates
from utils.eval import AverageMeterSet
from utils.metrics import write_metrics
from utils.misc import load_state, save_state

MIN_VALIDATION_SIZE = 50

logger = logging.getLogger()


class MixMatchTransform:
    """
    MixMatchTransform implements the augmentation strategy as proposed by MixMatch, which - depending on the context -
    returns multiple augmented versions of a single image.
    """
    def __init__(self, transform: Callable, k: int):
        """
        Initializes a MixMatchTransform object.

        Parameters
        ----------
        transform: Callable
            The augmentation strategy usually given by a torchvision.transform object
        k: int
            Number of augmented versions of given image to return. If K > 1 a list of augmented versions of
            an image is returned.
        """
        self.K = k
        self.transform = transform

    def __call__(self, img: Image):
        """
        Applies augmentation strategy specified by self.transform to input image.

        Parameters
        ----------
        img: PIL.Image
            Input image for which augmented version(s) are computed.

        Returns
        -------
        transformed_image: Optional[PIL.Image, List]
            Returns augmented version(s) of the input image.
        """
        if self.K <= 1:
            return self.transform(img)
        else:
            return [self.transform(img) for _ in range(self.K)]

    @classmethod
    def default(cls, dataset: str, k: int, img_size: int = 32, padding: int = 4):
        """
        Default constructor for the MixMatchTransform class which uses the augmentation strategy
        proposed in MixMatch. The weak augmentation strategy consists of random horizontal flips, random crops and
        a normalization operation.

        Parameters
        ----------
        cls: MixMatchTransform
            Reference to the MixMatchTransform class
        dataset: str
            String specifying dataset to which transform is applied. Important to select correct normalizer.
        k: int
            Number of augmented versions of input image to return
        img_size: int
            Size of input images (assuming images are squared)
        padding: int
            Number of padding pixels used as input to weak_augmentation transform
        Returns
        -------
        cls: MixMatchTransform
            Function returns instance of MixMatchTransform based on given inputs.
        """
        return cls(
            transforms.Compose(
                [
                    get_weak_augmentation(img_size, padding),
                    get_normalizer(dataset)
                ]
            ),
            k
        )


def get_transform_dict(args):
    """
    Generates dictionary with transforms for all datasets

    Parameters
    ----------
    args: argparse.Namespace
        Namespace object that contains all command line arguments with their corresponding values
    Returns
    -------
    transform_dict: Dict
        Dictionary containing transforms for the labeled train set, unlabeled train set
        and the validation / test set
    """
    img_size = IMG_SIZE[args.dataset]
    padding = int(
        0.125 * img_size
    )  # default value is to choose padding size as 12.5% of image size
    return {
        "train": MixMatchTransform.default(
            args.dataset, k=1, img_size=img_size, padding=padding
        ),
        "train_unlabeled": MixMatchTransform.default(
            args.dataset, k=args.num_augmentations, img_size=img_size, padding=padding
        ),
        "test": get_normalizer(args.dataset),
    }


def get_optimizer(args, model: torch.nn.Module):
    """
    Initialize and return Adam optimizer

    Parameters
    ----------
    args: argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values
    model: torch.nn.Module
        torch module which is trained using MixMatch
    Returns
    -------
    optim: torch.optim.Optimizer
        Returns adam optimizer which is used for model training
    """
    return Adam(model.parameters(), lr=args.lr)


def train(
        args,
        model: torch.nn.Module,
        train_loader_labeled: DataLoader,
        train_loader_unlabeled: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        writer: SummaryWriter,
        **kwargs
):
    """
    Method for MixMatch training of model based on given data loaders and parameters.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values
    model: torch.nn.Module
        The torch model to train
    train_loader_labeled: DataLoader
        Data loader of labeled dataset
    train_loader_unlabeled: DataLoader
        Data loader of unlabeled dataset
    validation_loader: DataLoader
        Data loader of validation set (by default MixMatch does not use a validation dataset)
    test_loader: DataLoader
        Data loader of test set
    writer: SummaryWriter
        SummaryWriter instance which is used to write losses as well as training / evaluation metrics
        to tensorboard summary file.
    Returns
    -------
    model: torch.nn.Module
        The method returns the trained model
    ema_model: EMA
        The EMA class which maintains an exponential moving average of model parameters. In MixMatch the exponential
        moving average parameters are used for model evaluation and for the reported results.
    writer: SummaryWriter
        SummaryWriter instance which is used to write losses as well as training / evaluation metrics
        to tensorboard summary file.
    """
    model.to(args.device)

    if args.use_ema:
        ema_model = EMA(model, args.ema_decay)
    else:
        ema_model = None

    optimizer = get_optimizer(args, model)

    best_acc = 0
    start_epoch = 0

    # If training is resumed - load saved state dict (optimizer, model, ema_model, ...)
    if args.resume:
        checkpoint_file = next(
            filter(
                lambda x: x.endswith(".tar"),
                sorted(os.listdir(args.resume), reverse=True),
            )
        )
        state_dict = load_state(os.path.join(args.resume, checkpoint_file))
        model.load_state_dict(state_dict["model_state_dict"])
        if args.use_ema:
            ema_model.shadow = state_dict["ema_model_shadow"]
        if optimizer:
            optimizer.load_state_dict(state_dict["optimizer"])
        best_acc = state_dict["acc"]
        start_epoch = state_dict["epoch"]

    for epoch in range(start_epoch, args.epochs):
        train_total_loss, train_labeled_loss, train_unlabeled_loss = train_epoch(
            args,
            model,
            ema_model,
            train_loader_labeled,
            train_loader_unlabeled,
            optimizer,
            epoch
        )

        if args.use_ema:
            ema_model.assign(model)
            val_metrics = evaluate(args, validation_loader, model, epoch, "Validation")
            test_metrics = evaluate(args, test_loader, model, epoch, "Test")
            ema_model.resume(model)
        else:
            val_metrics = evaluate(args, validation_loader, model, epoch, "Validation")
            test_metrics = evaluate(args, test_loader, model, epoch, "Test")

        writer.add_scalar("Loss/train_total", train_total_loss, epoch)
        writer.add_scalar("Loss/train_labeled", train_labeled_loss, epoch)
        writer.add_scalar("Loss/train_unlabeled", train_unlabeled_loss, epoch)
        write_metrics(writer, epoch, val_metrics, descriptor="val")
        write_metrics(writer, epoch, test_metrics, descriptor="test")
        writer.flush()

        # Only save best model (based on validation accuracy) if validation set is sufficiently large
        # MixMatch does not use a validation set by default - so this will usually not be saved.
        if val_metrics.top1 > best_acc and args.save and len(validation_loader.dataset) > MIN_VALIDATION_SIZE:
            save_path = kwargs.get("save_path", args.out_dir)
            save_state(
                epoch,
                model,
                val_metrics.top1,
                optimizer,
                None,  # No LR-scheduler used in MixMatch
                ema_model,
                save_path,
                filename="best_model.tar",
            )
            best_acc = val_metrics.top1

        # Save checkpoints during model training at a fixed interval specified by args.checkpoint_interval
        if epoch % args.checkpoint_interval == 0 and args.save:
            save_path = kwargs.get("save_path", args.out_dir)
            old_checkpoint_files = list(
                filter(lambda x: "checkpoint" in x, os.listdir(save_path))
            )
            save_state(
                epoch,
                model,
                val_metrics.top1,
                optimizer,
                None,  # No LR-scheduler used in MixMatch
                ema_model,
                save_path,
                filename=f"checkpoint_{epoch}.tar",
            )

            # Delete old checkpoint files in order to save space
            for file in old_checkpoint_files:
                os.remove(os.path.join(save_path, file))

    writer.close()
    logger.info(
        "Finished MixMatch training: Validation: Acc@1 {val_acc1:.3f}\tAcc@5 {val_acc5:.3f}\t Test: Acc@1 {test_acc1:.3f} Acc@5 {test_acc5:.3f}".format(
            val_acc1=val_metrics.top1,
            val_acc5=val_metrics.top5,
            test_acc1=test_metrics.top1,
            test_acc5=val_metrics.top5,
        )
    )
    save_path = kwargs.get("save_path", args.out_dir)
    if args.save:
        old_checkpoint_files = list(
            filter(lambda x: "checkpoint" in x, os.listdir(save_path))
        )
        for file in old_checkpoint_files:
            os.remove(os.path.join(save_path, file))

    save_state(
        epoch,
        model,
        val_metrics.top1,
        optimizer,
        None,  # No scheduler used in MixMatch implementation
        ema_model,
        save_path,
        filename="last_model.tar",
    )
    return model, ema_model, writer


def interleave_offsets(batch_size: int, nu: int):
    """
    Method computing offsets that are used to interleave batches in MixMatch.
    Parameters
    ----------

    Returns
    -------
    offsets: List
        List of length (nu+1), which specifies offsets used for interleaving batches. For example, offsets in MixMatch
        are of length 4 and in the default confguration are given by [0, 21, 42, 64].
    """
    groups = [batch_size // (nu + 1)] * (nu + 1)
    for x in range(batch_size - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch_size
    return offsets


def interleave(xy: List, batch_size: int):
    """
    Reference: https://github.com/google-research/mixmatch
    Method that interleaves both labeled and unlabeled batches. This is necessary to ensure that the BatchNorm update,
    which is only performed once per iteration for the first batch in xy (see lines 531-533), is representative
    of the entire dataset.

    If xy contains three tensors x (x_0:x_5), y (y_0:y_5) and z (z_0:z_5) with a batch size of 6, the offset
    List will be [0, 2, 4, 6]. Interleave will then perform the following substitutions:
        - elements x_0:x_1 (offsets[0]:offsets[1]) will remain unchanged
        - elements x_2:x_3 (offsets[1]:offsets[2]) will be exchanged with y_2:y_3
        - elements x_4:x_5 (offsets[2]:offsets[4]) will be exchanged with z_4:z_5

    This is also illustrated in the following:
        x_0 , y_0 , z_0               x_0 , y_0 , z_0
        x_1 , y_1 , z_1               x_1 , y_1 , z_1
        x_2 , y_2 , z_2    ---- >     y_2 , x_2 , z_2
        x_3 , y_3 , z_3    ---- >     y_3 , x_3 , z_3
        x_4 , y_4 , z_4               z_4 , y_4 , x_4
        x_5 , y_5 , z_5               z_5 , y_5 , x_5

    This ensures that the first tensor of the returned list, i.e. [x_0, x_1, y_2, y_3, z_4, z_5] contains images that
    represent the entire data distribution in order to perform the correct BatchNorm update at every iteration.

    The same method can then be used to reverse the substitutions after all batches have been passed through the model.

    Parameters
    ----------
    xy: List[torch.Tensor]
        List of tensors which should be interleaved. In MixMatch this list is generally of length 3:
        [labeled_batch (NxCxHxW), unlabeled_batch_aug_1 (NxCxHxW), unlabeled_batch_aug_2 (NxCxHxW)].
    batch_size: int
        Batch size, i.e. first tensor dimension, of the tensors in the list xy
    Returns
    -------
    interleaved: List
        List of interleaved tensors as described above.
    """
    nu = len(xy) - 1
    offsets = interleave_offsets(batch_size, nu)
    xy = [[v[offsets[p]: offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def mixup(all_inputs: torch.Tensor, all_targets: torch.Tensor, alpha: float):
    """
    Implements MixUp operation (https://arxiv.org/abs/1710.09412) between given samples and targets.
    The mixing parameter (lambda) is sampled from a beta distribution for every pair of samples.

    Parameters
    ----------
    all_inputs: torch.Tensor
        Tensor composed of a concatenation of batches of labeled and unlabeled training samples at the current iteration.
    all_targets: torch.Tensor´
        Tensor composed of concatenation of batches of labeled and unlabeled training samples at the current iteration.
    alpha: float
        Shape parameter for the beta distribution used for sampling the mixing parameters. The mixing parameters are
        sampled from Beta(alpha, alpha).
    Returns
    -------
    mixed_inputs: torch.Tensor
        Tensor containing the mixed samples which are computed based on input samples
    mixed_targets: torch.Tensor
        Tensor containing the mixed labels of samples in mixed_inputs
    """
    mixup_lambda = np.random.beta(alpha, alpha, size=all_inputs.size()[0])
    mixup_lambda = torch.tensor(np.maximum(mixup_lambda, 1 - mixup_lambda), dtype=torch.float32).to(all_inputs.device)

    idx = torch.randperm(all_inputs.size()[0])

    original_input, shuffled_input = all_inputs, all_inputs[idx]
    original_targets, shuffled_targets = all_targets, all_targets[idx]

    mixed_inputs = mixup_lambda.view(-1, 1, 1, 1) * original_input + \
                  (1 - mixup_lambda.view(-1, 1, 1, 1)) * shuffled_input
    mixed_targets = mixup_lambda.view(-1, 1) * original_targets + \
                    (1 - mixup_lambda.view(-1, 1)) * shuffled_targets
    return mixed_inputs, mixed_targets


def train_epoch(
        args,
        model: torch.nn.Module,
        ema_model: EMA,
        train_loader_labeled: torch.utils.data,
        train_loader_unlabeled: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
):
    """
    Method that executes a training epoch, i.e. a pass through all train samples in the training data loaders.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace with command line arguments and corresponding values
    model: torch.nn.Module
        The method returns the trained model
    ema_model: EMA
        The EMA class which maintains an exponential moving average of model parameters. In MixMatch the exponential
        moving average parameters are used for model evaluation and for the reported results.
    train_loader_labeled: DataLoader
        Data loader fetching batches from the labeled set of data.
    train_loader_unlabeled: DataLoader
        Data loader fetching batches from the unlabeled set of data.
    optimizer: Optimizer
        Optimizer used for model training. In the case of MixMatch this is an Adam optimizer.
    epoch: int
        Current epoch
    Returns
    -------
    train_stats: Tuple
        The method returns a tuple containing the total, labeled and unlabeled loss.
    """
    meters = AverageMeterSet()

    model.zero_grad()
    model.train()

    if args.pbar:
        p_bar = tqdm(range(len(train_loader_labeled)))

    for batch_idx, batch in enumerate(
            zip(train_loader_labeled, train_loader_unlabeled)
    ):
        loss = train_step(
            args, model, batch, meters, epoch=epoch, batch_idx=batch_idx
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA model if configured
        if args.use_ema:
            ema_model(model)

        # Apply weight decay to current model
        apply_wd(model, args.wd)

        if args.pbar:
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. | Lambda U: {wu:.3f}".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(train_loader_labeled),
                    wu=meters["wu"].val,
                )
            )
            p_bar.update()

    if args.pbar:
        p_bar.close()
    return (
        meters["total_loss"].avg,
        meters["labeled_loss"].avg,
        meters["unlabeled_loss"].avg,
    )


def train_step(args, model: torch.nn.Module, batch: Tuple, meters: AverageMeterSet, epoch: int, batch_idx: int):
    """
    Method that executes a MixMatch training step, i.e. a single training iteration.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace with command line arguments and corresponding values
    model: torch.nn.Module
        The method returns the trained model
    batch: Tuple
        Tuple containing loaded objects from both labeled and unlabeled data loaders. Each object is another tuple
        containing samples and targets (only of labeled batch).
    meters: AverageMeterSet
        AverageMeterSet object which is used to track training and testing metrics (loss, accuracy, ...)
        over the entire training process.
    epoch: int
        Current epoch
    batch_idx: int
        Current batch_idx, i.e. iteration step
    Returns
    -------
    loss: torch.Tensor
        Tensor containing the total MixMatch loss (attached to computational graph) used for optimization
        by backpropagation.
    """
    labeled_batch, unlabeled_batch = batch
    labeled, targets = labeled_batch
    unlabeled_k, _ = unlabeled_batch

    # One hot labels
    targets = torch.zeros(args.batch_size, args.num_classes).scatter_(
        1, targets.view(-1, 1), 1
    )

    unlabeled_k = [u_k.to(args.device) for u_k in unlabeled_k]
    labeled = labeled.to(args.device)
    targets = targets.to(args.device)

    # Disable batch-norm running_mean and running_var updates for pseduo-label forward passes
    set_bn_running_updates(model, enable=False)
    with torch.no_grad():
        preds = [
            torch.softmax(model(u_k.to(args.device)), dim=1) for u_k in unlabeled_k
        ]
        avg_preds = torch.stack(preds).mean(dim=0)
        sharpened_preds = torch.pow(avg_preds, 1 / args.temperature)
        unlabeled_targets = sharpened_preds / sharpened_preds.sum(dim=-1, keepdim=True)
        unlabeled_targets = unlabeled_targets.detach()

    all_inputs = torch.cat([labeled] + unlabeled_k, dim=0)
    all_targets = torch.cat(
        [targets] + [unlabeled_targets for _ in range(len(unlabeled_k))], dim=0
    )

    mixed_input, mixed_targets = mixup(all_inputs, all_targets, args.alpha)

    # Interleave labeled and unlabeled samples to avoid biased batch norm calculation
    mixed_input = list(torch.split(mixed_input, args.batch_size))
    mixed_input = interleave(mixed_input, args.batch_size)

    # Only update running batch-norm parameters for first batch of mixed batches
    set_bn_running_updates(model, enable=True)
    logits = [model(mixed_input[0])]
    set_bn_running_updates(model, enable=False)
    for input in mixed_input[1:]:
        logits.append(model(input))

    # Put interleaved samples back - reverses interleaving applied before
    logits = interleave(logits, args.batch_size)
    logits_x = logits[0]
    logits_u = torch.cat(logits[1:], dim=0)

    # Cross entropy loss for labeled samples
    labeled_loss = -torch.sum(
        F.log_softmax(logits_x, dim=1) * mixed_targets[: args.batch_size], dim=1
    )
    # L2-distance loss for unlabeled samples
    unlabeled_loss = torch.mean(
        (torch.softmax(logits_u, dim=1) - mixed_targets[args.batch_size :]) ** 2
    )

    # Update unlabeled loss weight based on current step (linear rampup to max. value over first 16 epochs)
    step = epoch * args.iters_per_epoch + (batch_idx + 1)
    wu = (
        args.wu * linear_rampup(step, 16 * args.iters_per_epoch)
        if not args.resume
        else args.wu
    )

    # Total loss
    loss = torch.mean(labeled_loss) + wu * unlabeled_loss

    meters.update("total_loss", loss.item(), targets.size(0))
    meters.update("labeled_loss", torch.mean(labeled_loss).item(), targets.size(0))
    meters.update("unlabeled_loss", unlabeled_loss.item(), targets.size(0))
    meters.update("wu", wu, 1)
    return loss
