import json
import pathlib
import time
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
from ..datasets import train_val_datasets
from ..metrics import model_size, correct, memory_size
from ..util import CSVLogger

from ..strategies import *
shortrepr = {
    RandomPruning: 'grnd',
    LayerMagnitudePruning: 'lmag',
    GlobalMagnitudePruning: 'gmag',
    LayerGradMagnitudePruning: 'lmagd',
    GlobalGradMagnitudePruning: 'gmagd',
    LayerActivationMagnitudePruning: 'lmaga',
    GlobalActivationMagnitudePruning: 'gmaga',
    ActivationNormChannelPruning: 'achn',
    WeightNormChannelPruning: 'wchn',
}

RESULTS_DIR = pathlib.Path('../results')
TENSORBOARD_DIR = pathlib.Path('../tblogs')


class PruningExperiment:

    def __init__(self,
                 dataset,
                 model,
                 strategy,
                 compression,
                 seed=42,
                 path=None,
                 dl_kwargs=tuple(),
                 train_kwargs=tuple(),
                 debug=False,
                 pretrained=True,
                 resume=None,
                 resume_optim=False):

        # Data loader params
        # TODO Look into why pin_memory seems to decrease performance for all data-model combinations
        self.dl_kwargs = {'batch_size': 128,
                          'pin_memory': False,
                          'num_workers': 8
                          }
        self.dl_kwargs.update(dl_kwargs)

        self.train_kwargs = {'optim': 'SGD',
                             'epochs': 30,
                             'lr': 1e-3,
                             }
        self.train_kwargs.update(train_kwargs)

        # Log all params for reproducibility
        # TODO Fix the locals() so that it does not produce '"foobar"' strings
        params = {k: repr(v) for k, v in locals().items() if k != 'self'}
        params['dl_kwargs'] = self.dl_kwargs
        params['train_kwargs'] = self.train_kwargs
        self.params = params

        ############### PRUNING STRATEGY ###############
        if strategy is not None:
            # Load strategy
            self.compression = compression
            if isinstance(strategy, str):
                strategy = getattr(strategies, strategy)(compression)
            self.strategy = strategy
        else:
            assert compression == 1
            self.strategy = None
            self.compression = 1

        ############### DATASET ###############

        # Get dataset & dataloaders
        self.dataset = dataset
        self.train_dataset, self.val_dataset = train_val_datasets(dataset, preproc=True)
        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **self.dl_kwargs)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **self.dl_kwargs)

        ############### LOAD MODEL ###############

        # Load model
        self.model_name = model
        if isinstance(model, str):
            if hasattr(models, model):
                model = getattr(models, model)(pretrained=pretrained)

            elif hasattr(torchvision.models, model):
                # https://pytorch.org/docs/stable/torchvision/models.html
                model = getattr(torchvision.models, model)(pretrained=pretrained)
            else:
                raise ValueError(f"Model {model} not available in custom models or torchvision models")
        self.model = model
        self.pretrained = pretrained

        strat = shortrepr[strategy.__class__] if self.strategy is not None else "train"
        # Experiment name and folder
        self.name = f"{self.dataset}_" + \
                    f"{self.model_name}_" + \
                    f"{strat}_" + \
                    f"Z{self.compression}_" + \
                    f"R{seed}_" + \
                    f"{uid()}"

        if path is None:
            path = RESULTS_DIR
            if debug:
                path /= 'debug'
        self.path = pathlib.Path(path) / self.name

        # Resume weights
        self.resume = resume
        self.resume_optim = resume_optim
        if self.resume is not None:
            self.resume = pathlib.Path(self.resume)


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
        #tb_logdir = TENSORBOARD_DIR / self.name
        #self.tb_writer = SummaryWriter(log_dir=tb_logdir.as_posix())

        # CSV logs
        self.csvlogger = CSVLogger(self.path / 'finetuning.csv',
            ['epoch',
             'train_loss', 'train_acc1', 'train_acc5',
             'val_loss', 'val_acc1', 'val_acc5',
             'timestamp'])

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

        # Resuming from previous experiment
        if self.resume is not None:
            previous = torch.load(self.resume)
            if self.resume_optim:
                self.model.load_state_dict(previous['model_state_dict'])
                self.optim.load_state_dict(previous['optim_state_dict'])
            else:
                self.model.load_state_dict(previous)

        # TODO: as of now, when resuming, strategy must be set to none and compression too
        # Differentitate between --resume and --weights!

        #### Pruning ####
        # Prune it based on strategy
        if self.strategy is not None:
            print("Masking model")
            inputs, outputs = next(iter(self.train_dl))
            self.strategy.apply(self.model, inputs, outputs)
            printc("Masked model", color='GREEN')

        # Torch CUDA config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            printc("GPU NOT AVAILABLE, USING CPU!", color="RED")
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
        since = time.time()
        try:
            for epoch in range(1, self.epochs+1):
                printc(f"Start epoch {epoch}", color='YELLOW')
                self.train(epoch)
                self.eval(epoch)
                # TODO Early stopping
                # TODO ReduceLR on plateau?
                self.csvlogger.set(epoch=epoch)
                self.csvlogger.set(timestamp=time.time()-since)
                self.csvlogger.update()

                # Checkpoint epochs
                # TODO Model checkpointing based on best val loss/acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict()
                }, self.path / f'checkpoint-{epoch}.pt')

        except KeyboardInterrupt:
            printc(f"Interrupted at epoch {epoch}", color='RED')
            # TODO, allow SIGINT to edit num_epochs

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
        #self.tb_writer.close()
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
        # TODO check loss is right
        total_loss /= len(dl) * dl.batch_size
        acc1 /= len(dl) * dl.batch_size
        acc5 /= len(dl) * dl.batch_size

        # Tensorboard logging
        #self.tb_writer.add_scalar(f'{prefix} Loss', loss, epoch)
        #self.tb_writer.add_scalar(f'{prefix} Acc1', acc1, epoch)
        #self.tb_writer.add_scalar(f'{prefix} Acc5', acc5, epoch)

        self.csvlogger.set(**{
            f'{prefix.lower()}_loss': loss,
            f'{prefix.lower()}_acc1': acc1,
            f'{prefix.lower()}_acc5': acc5,
        })

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
        x, y = next(iter(self.val_dl))
        x, y = x.to(self.device), y.to(self.device)
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
               f"compression={self.compression}, " + \
               f"dataset={self.dataset}, " + \
               f"model={self.model_name}, " + \
               f"dl_kwargs={repr(self.dl_kwargs)}, " + \
               f"train_kwargs={repr(self.train_kwargs)}, " + \
               f"seed={self.seed}, " + \
               f"path={self.path.as_posix()}, " + \
               f"pretrained={self.pretrained}, " + \
               f"resume={self.resume}, " + \
               f"resume_optim={self.resume_optim}" + \
               f")"

    # TODO TO LOAD PREVIOUS EXPS
    # def frompath()
