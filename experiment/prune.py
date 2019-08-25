import csv
import json
import pathlib
from tqdm import tqdm

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

from alfred import printc, uid, color  # Color pretty printing
from pylot.util.summary import summary

# flor imports
from . import fix_seed
from .. import strategies
from .. import models
from ..datasets import get_datasets
from ..metrics import model_size, correct
from ..pruning import masked_module, compute_masks, apply_masks

RESULTS_DIR = pathlib.Path('../results')
TENSORBOARD_DIR = pathlib.Path('../tblogs')


class PruningExperiment:

    def __init__(self,
                 strategy,
                 pruning,
                 dataset,
                 model,
                 seed=42,
                 path=None,
                 dl_kwargs=tuple(),
                 train_kwargs=tuple()):

        # Data loader params
        self.dl_kwargs = {'batch_size': 32,
                          'pin_memory': True,
                          'num_workers': 6
                          }
        self.dl_kwargs.update(dl_kwargs)

        self.train_kwargs = {'optim': 'SGD',
                             'epochs': 50,
                             'lr': 1e-3,
                             }
        self.train_kwargs.update(train_kwargs)

        ############### REPRODUCIBILITY ###############

        # Log all params for reproducibility
        params = {k: repr(v) for k, v in locals().items() if k != 'self'}
        params['dl_kwargs'] = self.dl_kwargs
        params['train_kwargs'] = self.train_kwargs
        self.params = params

        # Fix python, numpy, torch seeds for reproducibility
        self.seed = seed
        fix_seed(self.seed)

        ############### PRUNING STRATEGY ###############

        # Load strategy
        self.pruning = pruning
        if isinstance(strategy, str):
            strategy = getattr(strategies, strategy)(pruning)
        self.strategy = strategy

        ############### DATASET ###############

        # Get dataset & dataloaders
        self.dataset = dataset
        self.train_dataset, self.val_dataset = get_datasets(dataset, preproc=True)
        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **self.dl_kwargs)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **self.dl_kwargs)

        ############### LOAD MODEL ###############

        # Load model
        self.model_name = model
        if isinstance(model, str):
            if hasattr(models, model):
                model = getattr(models, model)
            elif hasattr(torchvision.models, model):
                model = getattr(torchvision.models, model)(pretrained=True)
            else:
                raise ValueError(f"Model {model} not available in custom models or torchvision models")
        self.model = model

        # Experiment name and folder
        self.name = f"{self.model_name}_" + \
                    f"{self.dataset}_" + \
                    f"{self.strategy.shortrepr()}_" + \
                    f"R{self.seed}_" + \
                    f"{uid()}"

        if path is None:
            path = RESULTS_DIR / self.name
        self.path = pathlib.Path(path)



    def run(self):
        exp = repr(self) #.replace(', ', ',\n\t')
        printc(f"Running experiment {exp}", color='YELLOW')

        ## Logging init
        printc(f"Logging results to {self.path}", color='MAGENTA')

        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / 'params.json', 'w') as f:
            json.dump(self.params, f, indent=4)

        # Tensorboard logs
        tb_logdir = TENSORBOARD_DIR / self.expname
        self.tb_writer = SummaryWriter(log_dir=tb_logdir.as_posix())

        # CSV logs
        self.csv_file = open(self.path / 'finetuning.csv', 'w')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Train Loss',
                                  'Train Acc1',
                                  'Train Acc5',
                                  'Val Loss',
                                  'Val Acc1',
                                  'Val Acc5'])

        ### Training setup
        self.epochs = self.train_kwargs['epochs']
        optim = self.train_kwargs['optim']
        lr = self.train_kwargs['lr']
        if optim == 'SGD':
            self.optim = torch.optim.SGD(self.model.parameters(),
                                         lr,
                                         momentum=0.9,
                                         nesterov=True)
        elif optim == 'Adam':
            self.optim = torch.optim.Adam(self.model.parameters(), lr)

        self.loss_func = nn.CrossEntropyLoss()

        #### Pre-pruning ####
        # Get model, compute metrics before pruning

        #### Pruning ####
        # Prune it based on strategy
        print("Masking model")
        masked_module(self.model, self.strategy)
        masks = compute_masks(self.model, self.strategy)
        apply_masks(self.model, masks)
        printc("Masked model", color='GREEN')

        # Torch CUDA config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        cudnn.benchmark = True   # For fast training.

        with open(self.path / 'summary.txt', 'w') as f:
            # TODO Make size parametric of dataset
            print(summary(self.model, (3, 224, 224)), file=f)

        #### Pre-finetuning ####
        # Compute metrics and save them
        metrics = {}
        metrics['pre'] = self.compute_metrics()
        print(metrics['pre'])
        for m, v in metrics['pre'].items():
            print(m, type(v))
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        #### Finetuning ####
        for epoch in range(self.epochs):
            train_results = self.train(epoch)
            val_results = self.eval(epoch)
            # TODO Early stopping
            # TODO ReduceLR on plateau?
            self.csv_writer.writerow(train_results+val_results)
            self.csv_file.flush()

        #### Post-finetuning ####
        # Save Model
        model_path = self.path / 'model.pt'
        torch.save(self.model.state_dict(), model_path.as_posix())

        # Recompute metrics and save them
        metrics['post'] = self.compute_metrics()
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        # Logging Teardown
        self.tb_writer.close()
        self.csv_writer.close()

    def run_epoch(self, train, epoch=0):
        if train:
            prefix = 'Train'
            self.model.train()
        else:
            prefix = 'Val'
            self.model.eval()

        total_loss = 0
        acc1 = 0
        acc5 = 0

        dl = self.train_dl if train else self.val_dl
        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{prefix} Epoch {epoch}/{self.epochs}")

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)
                if train:
                    loss.backward()

                    self.optim.step()
                    self.optim.zero_grad()

                total_loss += loss.item()
                c1, c5 = correct(yhat, y, (1, 5))
                acc1 += c1
                acc5 += c5

                epoch_iter.set_postfix(loss=total_loss / i, #* dl.batch_size,
                                       top1=acc1.item() / (i * dl.batch_size),
                                       top5=acc5.item() / (i * dl.batch_size))

        total_loss /= len(dl) * dl.batch_size
        acc1 /= len(dl)
        acc5 /= len(dl)

        # Tensorboard logging
        self.tb_writer.add_scalar(f'{prefix} Loss', loss, epoch)
        self.tb_writer.add_scalar(f'{prefix} Acc1', acc1, epoch)
        self.tb_writer.add_scalar(f'{prefix} Acc5', acc5, epoch)

        # TODO Model checkpointing based on best val loss/acc

        return loss, acc1, acc5

    def train(self, epoch=0):
        return self.run_epoch(True, epoch)

    def eval(self, epoch=0):
        return self.run_epoch(False, epoch)

    def compute_metrics(self):

        metrics = {}
        # Model Size
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = size / size_nz

        # Memory Footprint
        # TODO
        memory, memory_nz = -1, -1
        metrics['memory'] = memory
        metrics['memory_nz'] = memory_nz

        # FLOPS
        # TODO
        flops, flops_nz = -1, -1
        metrics['flops'] = flops
        metrics['flops_nz'] = flops_nz

        # Accuracy
        acc1 = 0
        acc5 = 0
        with torch.no_grad():
            for x, y in tqdm(self.val_dl):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                c1, c5 = correct(yhat, y, (1, 5))
                acc1 += c1
                acc5 += c5

        acc1 /= len(self.val_dl)
        acc5 /= len(self.val_dl)

        metrics['val_acc1'] = acc1.item()
        metrics['val_acc5'] = acc5.item()

        return metrics

    def __repr__(self):
        return f"PruningExperiment(" + \
               f"strategy={self.strategy.__class__.__name__}, " + \
               f"pruning={self.pruning}, " + \
               f"dataset={self.dataset}, " + \
               f"model={self.model_name}, " + \
               f"dl_kwargs={repr(self.dl_kwargs)}, " + \
               f"train_kwargs={repr(self.train_kwargs)}, " + \
               f"seed={self.seed}, " + \
               f"path={self.path.as_posix()}" + \
               f")"

    # TODO TO LOAD PREVIOUS EXPS
    # def frompath()