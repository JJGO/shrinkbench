# ShrinkBench

Open source PyTorch library to facilitate development and  standardized evaluation of neural network pruning methods.

![](https://shrinkbench.github.io/diagram.svg)

## Paper

This repo contains the analysis and benchmarks results from the paper [What is the State of Neural Network Pruning?](https://arxiv.org/abs/2003.03033).


# Installation

First, install the dependencies, this repo depends on
 - `PyTorch`
 - `Torchvision`
 - `NumPy`
 - `Pandas`
 - `Matplotlib`

To install the dependencies

```bash
# Create a python virtualenv or conda env as necessary

# With conda
conda install numpy matplotlib pandas tqdm
conda install pytorch torchvision -c pytorch

# With pip
pip install numpy matplotlib pandas torch torchvision tqdm
```

then, to install the module itself you just need to clone the repo and  add the parent path it to your `PYTHONPATH`. For example:

```bash
git clone git@github.com:JJGO/shrinkbench.git shrinkbench

# Bash
echo "export PYTHONPATH=\"$PWD:\$PYTHONPATH\"" >> ~/.bashrc

# ZSH
echo "export PYTHONPATH=\"$PWD:\$PYTHONPATH\"" >> ~/.zshrc
```

# Strategies

ShrinkBench not only faciliates evaluation of pruning methods, but also their development. Here's the code for a simple implementation of Global Magnitude Pruning and Layerwise Magnitude Pruning. As you can see, it is quite succint; you are just tasked with implementing `model_masks` a function that returns the masks for the model's weight tensors. If you want to prune your model layerwise, then you just need to implement `layer_masks`. For more examples, see the source code for the provided baselines.

```python
class GlobalMagWeight(VisionPruning):

    def model_masks(self):
        importances = map_importances(np.abs, self.params())
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class LayerMagWeight(LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        importances = {param: np.abs(value) for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks
```

# Experiments

See [here](jupyter/experiment_tutorial.ipynb) for a notebook showing how to run pruning experiments and plot their results

## Modules

The modules are organized as follows:

| submodule | Description |
| ---- | ---- |
| `analysis/` | Aggregated survey results over 80 pruning papers |
| `datasets/` | Standardized dataloaders for supported datasets |
| `experiment/` | Main experiment class with the data loading, pruning, finetuning & evaluation |
| `metrics/` | Utils for measuring accuracy, model size, flops & memory footprint |
| `models/` | Custom architectures not included in `torchvision` |
| `plot/` | Utils for plotting across the logged dimensions |
| `pruning/` | General pruning and masking API.  |
| `scripts/` | Executable scripts for running experiments (see `experiment/`) |
| `strategies/` | Baselines pruning methods, mainly magnitude pruning based |

