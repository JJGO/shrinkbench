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

from alfred import printc, uid  # Misc utils
# from pylot.util.summary import summary

# flor imports
from . import fix_seed
from .. import strategies
from .. import models
from ..datasets import get_datasets
from ..metrics import model_size, correct
from ..pruning import mask_module, compute_masks
from ..util import CSVLogger

RESULTS_DIR = pathlib.Path('../results')
DEBUG_DIR = pathlib.Path('../debug')
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
                 train_kwargs=tuple(),
                 debug=False):

        # Data loader params
        self.dl_kwargs = {'batch_size': 32,
                          'pin_memory': True,
                          'num_workers': 4
                          }
        self.dl_kwargs.update(dl_kwargs)

        self.train_kwargs = {'optim': 'SGD',
                             'epochs': 50,
                             'lr': 1e-3,
                             }
        self.train_kwargs.update(train_kwargs)

        # Log all params for reproducibility
        params = {k: repr(v) for k, v in locals().items() if k != 'self'}
        params['dl_kwargs'] = self.dl_kwargs
        params['train_kwargs'] = self.train_kwargs
        self.params = params

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
                model = getattr(models, model)(pretrained=True)

            elif hasattr(torchvision.models, model):
                model = getattr(torchvision.models, model)(pretrained=True)
            else:
                raise ValueError(f"Model {model} not available in custom models or torchvision models")
        self.model = model

        # Experiment name and folder
        self.name = f"{self.model_name}_" + \
                    f"{self.dataset}_" + \
                    f"{self.strategy.shortrepr()}_" + \
                    f"R{seed}_" + \
                    f"{uid()}"

        if path is None:
            if not debug:
                path = RESULTS_DIR / self.name
            else:
                path = DEBUG_DIR / self.name
        self.path = pathlib.Path(path)

        ############### REPRODUCIBILITY ###############
        # Fix python, numpy, torch seeds for reproducibility
        self.seed = seed
        fix_seed(self.seed)

    def _pre_run(self):
        ## Logging init
        printc(f"Logging results to {self.path}", color='MAGENTA')

        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / 'params.json', 'w') as f:
            json.dump(self.params, f, indent=4)

        # Tensorboard logs
        tb_logdir = TENSORBOARD_DIR / self.name
        self.tb_writer = SummaryWriter(log_dir=tb_logdir.as_posix())

        # CSV logs
        self.csvlogger = CSVLogger(self.path / 'finetuning.csv',
            ['epoch',
             'train_loss', 'train_acc1', 'train_acc5',
             'val_loss', 'val_acc1', 'val_acc5'])

    def run(self):
        printc(f"Running {repr(self)}", color='YELLOW')

        self._pre_run()

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
        # masked_module(self.model, self.strategy)
        masks = compute_masks(self.model, self.strategy)
        mask_module(self.model, masks)
        # apply_masks(self.model, masks)
        printc("Masked model", color='GREEN')

        # Torch CUDA config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        cudnn.benchmark = True   # For fast training.

        #### Pre-finetuning ####
        # Compute metrics and save them
        metrics = {}
        metrics['pre'] = self.compute_metrics()
        printc(metrics['pre'], color='GRASS')
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        #### Finetuning ####
        try:
            for epoch in range(1, self.epochs+1):
                printc(f"Start epoch {epoch}", color='YELLOW')
                self.train(epoch)
                self.eval(epoch)
                # TODO Early stopping
                # TODO ReduceLR on plateau?
                self.csvlogger.set(epoch=epoch)
                self.csvlogger.update()
        except KeyboardInterrupt:
            printc(f"Interrupted at epoch {epoch}", color='RED')


        #### Post-finetuning ####
        # Save Model
        model_path = self.path / 'model.pt'
        torch.save(self.model.state_dict(), model_path.as_posix())

        # Recompute metrics and save them
        metrics['post'] = self.compute_metrics()
        printc(metrics['post'], color='GRASS')
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        self.metrics = metrics

        # Logging Teardown
        self.tb_writer.close()
        self.csvlogger.close()

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
        acc1 /= len(dl) * dl.batch_size
        acc5 /= len(dl) * dl.batch_size

        # Tensorboard logging
        self.tb_writer.add_scalar(f'{prefix} Loss', loss, epoch)
        self.tb_writer.add_scalar(f'{prefix} Acc1', acc1, epoch)
        self.tb_writer.add_scalar(f'{prefix} Acc5', acc5, epoch)

        self.csvlogger.set(**{
            f'{prefix.lower()}_loss': loss,
            f'{prefix.lower()}_acc1': acc1,
            f'{prefix.lower()}_acc5': acc5,
        })

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
        loss, acc1, acc5 = self.run_epoch(False, -1)

        metrics['loss'] = loss.item()
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