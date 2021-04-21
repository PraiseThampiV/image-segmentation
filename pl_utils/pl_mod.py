import torch
import pytorch_lightning as pl
from torch import optim
from torch.utils.data import DataLoader

class SegmModelPL(pl.LightningModule):
    def __init__(self, model, train_set, val_set, test_set, batch_size, test_batch_size, 
    loss_criterion:list, loss_sum_weights:list=None, val_acc_func:list=None, optimizer = optim.Adam,
    lr=1e-4, experiment_name=""):
        """

        Parameters
        ----------
        model : `torch.nn.module`
            model
        train_set : `torch.utils.data.dataset.Dataset`
            train set
        val_set : `torch.utils.data.dataset.Dataset`
            validation dataset
        test_set : `torch.utils.data.dataset.Dataset`
            test dataset
        batch_size : int
            batch size
        test_batch_size : int
            test batch size
        loss_criterion : list
            list of losses
        loss_sum_weights : list, optional
            weights for each loss, by default None
        val_acc_func : list, optional
            list of accuracy functions, by default None
        optimizer : `torch.optim`, optional
            optimizer, by default optim.Adam
        lr : float, optional
            learning rate, by default 1e-4
        experiment_name : str, optional
            name of expt, by default ""
        """
        super(SegmModelPL, self).__init__()
        self.model = model
        self.train_set = train_set 
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size 
        self.test_batch_size = test_batch_size
        self.loss_criterion = loss_criterion
        self.loss_sum_weights = loss_sum_weights
        self.val_acc_func = val_acc_func
        self.optimizer = optimizer
        self.lr = lr
        self.experiment_name = experiment_name #model+loss names generally
        htc_gpu=torch.cuda.device_count()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kwargs = {'num_workers': 4*htc_gpu, 'pin_memory': True} if device == 'cuda' else {}

    def forward(self, x):
            out = self.model(x)
            return out
    def prepare_data(self):
        pass
    def training_step(self, batch, batch_nb):
        img, segm = batch
        pred_segm = self.forward(img)
        aggr_loss = 0
        for loss_ind, loss in enumerate(self.loss_criterion):
            loss_each = loss(pred_segm, segm)
            if self.loss_sum_weights is not None:
                loss_each = self.loss_sum_weights[loss_ind]*loss_each
            aggr_loss+=loss_each
        tensorboard_logs = {f'train_loss_{self.experiment_name}': aggr_loss}
        return {'loss': aggr_loss, 'log': tensorboard_logs}
    def validation_step(self, batch, batch_nb):
        img, segm = batch
        pred_segm = self.forward(img)
        aggr_loss = 0
        for loss_ind, loss in enumerate(self.loss_criterion):
            loss_each = loss(pred_segm, segm)
            if self.loss_sum_weights is not None:
                loss_each = self.loss_sum_weights[loss_ind]*loss_each
            aggr_loss+=loss_each
        
        val_acc = self.val_acc_func(pred_segm, segm)
        return {'val_loss_epoch': aggr_loss, 'val_acc_': val_acc}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss_epoch'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc_'] for x in outputs]).mean()
        tensorboard_logs = {f'avg_val_loss_{self.experiment_name}': avg_loss, f'val_acc_avg_{self.experiment_name}':avg_acc}
        return {'val_loss': avg_loss, 'val_acc':avg_acc, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        img, segm = batch
        pred_segm = self.forward(img)
        aggr_loss = 0
        for loss_ind, loss in enumerate(self.loss_criterion):
            loss_each = loss(pred_segm, segm)
            if self.loss_sum_weights is not None:
                loss_each = self.loss_sum_weights[loss_ind]*loss_each
            aggr_loss+=loss_each
        test_acc = self.val_acc_func(pred_segm, segm)
        return {'test_loss': aggr_loss, 'test_acc_': test_acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_avg_acc = torch.stack([x['test_acc_'] for x in outputs]).mean()
        logs = {f'test_loss_{self.experiment_name}': avg_loss, f'test_acc_{self.experiment_name}' : test_avg_acc}
        return {'test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return self.optimizer(self.parameters(),self.lr) ##.to(cuda)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, **self.kwargs)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, **self.kwargs)

    def test_dataloader(self):
           return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False, **self.kwargs)