# ShrinkBench

Open source PyTorch library to facilitate development and  standardized evaluation of neural network pruning methods. 

![](https://raw.githubusercontent.com/shrinkbench/shrinkbench.github.io/master/diagram.svg)

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

Requirements: 
 - `PyTorch`
 - `Torchvision`
 - `NumPy`
 - `Pandas`
 - `Matplotlib`
