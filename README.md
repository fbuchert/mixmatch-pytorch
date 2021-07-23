# PyTorch Implementation: MixMatch
PyTorch implementation of [MixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence
](https://arxiv.org/abs/1905.02249) based on the [official tensorflow implementation](https://github.com/google-research/mixmatch).

The implementation supports the following datasets:
- CIFAR-10 / CIFAR-100
- SVHN
- Caltech101 / Caltech256
- STL10
- HAM10000
- ImageNet



## Installation
Required python packages are listed in `requirements.txt`. All dependencies can be installed using pip
```
pip install -r requirements.txt
```
or using conda
```
conda install --file requirements.txt
```

## Training
MixMatch training (default configuration) is started by running the following command:
```
python main.py
```

### Configuration
All commandline arguments, which can be used to adapt the configuration of MixMatch are defined and described in `arguments.py`.
By default the following MixMatch configuration is run:
```
model: 'wide_resnet28_2'
dataset: 'cifar10'
lr: 0.002
wd: 0.00004
num_labeled: 250 (number of labeled samples, i.e. 25 labeled samples per class for cifar10)
epochs: 1024
iters_per_epoch: 1024
batch_size: 64
device: 'cuda'
out_dir: 'mixmatch'
mu: 1
temperature: 0.5
num_augmentations: 2
wu: 75
alpha: 0.75
```
In addition to these, the following arguments can be used to further configure the MixMatch training process:
  * `--device <cuda / cpu>`: Specify whether training should be run on GPU (if available) or CPU
  * `--num-workers <num_workers>`: Number of workers used by torch dataloader  
  * `--resume <path to run_folder>`: Resumes training of training run saved at specified path, e.g. `'out/mixmatch_training/run_0'`. Dataset splits, model state, optimizer state, etc.
   are loaded and training is resumed with specified arguments.

Alternatively, the `polyaxon.yaml`-file can be used to start MixMatch training on a polyaxon-cluster:
```
polyaxon run -f polyaxon.yaml -u
```
For a general introduction to polyaxon and its commandline client, please refer to the [official documentation](https://github.com/polyaxon/polyaxon)
## Monitoring
The training progress (loss, accuracy, etc.) can be monitored using tensorboard as follows:
```
tensorboard --logdir <result_folder>
```
This starts a tensorboard instance at `localhost:6006`, which can be opened in any common browser.

## Evaluation


## References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```