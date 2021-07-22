# MixMatch
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
MixMatch training is started by running the following command:
```
python main.py
```
All commandline arguments, which can be used to adapt the configuration of MixMatch are defined and described in `arguments.py`.


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