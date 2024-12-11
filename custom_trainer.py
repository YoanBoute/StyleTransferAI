from tqdm.auto import tqdm
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
import lightning as L
from pathlib import Path
import warnings

class EpochProgressBar(Callback):
    """Custom Progress Bar designed to replace Lightning's default progress bar, cycling on the total of epochs instead of on the number of steps per epoch

    Args:
        Callback (_type_): _description_
    """
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_bar = None

    def on_train_start(self, trainer, pl_module):
        self.epoch_bar = tqdm(total=self.total_epochs, desc="Training Progress", position=0, leave=True, dynamic_ncols=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_bar is not None:
            self.epoch_bar.update(1)
            
        loss = trainer.callback_metrics.get('General loss') 
        for logged_metric, value in trainer.callback_metrics.items() :
            self.epoch_bar.set_postfix({f"{logged_metric.capitalize().replace('_', ' ')}": value.item()})

    def on_train_end(self, trainer, pl_module):
        if self.epoch_bar is not None:
            self.epoch_bar.close()


class EmptyDataModule(L.LightningDataModule) :
    """Custom data module with not content to be used with Lightning Trainer on Style Transfer"""
    def __init__(self):
        super().__init__()
    
    def train_dataloader(self):
        return DataLoader([0], batch_size=1)
    

class CustomTrainer(L.Trainer) :
    """Custom Trainer for the Style Transfer task, which sets pertinent default values for this task. 
    In particular, the progress bar is replaced to cycle on the total of epochs rather than on the number of steps per epoch."""
    def __init__(self, *, accelerator = "auto", strategy = "auto", devices = "auto", num_nodes = 1, precision = None, logger = None, callbacks = None, fast_dev_run = False, max_epochs = None, min_epochs = None, max_steps = -1, min_steps = None, max_time = None, limit_train_batches = None, limit_val_batches = None, limit_test_batches = None, limit_predict_batches = None, overfit_batches = 0, val_check_interval = None, check_val_every_n_epoch = 1, num_sanity_val_steps = None, log_every_n_steps = None, enable_checkpointing = None, enable_progress_bar = None, enable_model_summary = None, accumulate_grad_batches = 1, gradient_clip_val = None, gradient_clip_algorithm = None, deterministic = None, benchmark = None, inference_mode = True, use_distributed_sampler = True, profiler = None, detect_anomaly = False, barebones = False, plugins = None, sync_batchnorm = False, reload_dataloaders_every_n_epochs = 0, default_root_dir = None):
        warnings.filterwarnings("ignore", category=UserWarning, message=".*num_workers.*")
        
        if enable_checkpointing is None :
            enable_checkpointing = True
        
        custom_progress_bar = EpochProgressBar(max_epochs)
        if callbacks is None :
            callbacks = [custom_progress_bar]
        else :
            callbacks.append(custom_progress_bar)
        
        if enable_progress_bar is None :
            enable_progress_bar = False
        
        if enable_model_summary is None :
            enable_model_summary = False
        
        if log_every_n_steps is None :
            log_every_n_steps = 1
        
        if default_root_dir is None :
            default_root_dir = Path('D:/StyleTransferAI/StyleTransferAI/trainings')
        
        super().__init__(accelerator=accelerator, strategy=strategy, devices=devices, num_nodes=num_nodes, precision=precision, logger=logger, callbacks=callbacks, fast_dev_run=fast_dev_run, max_epochs=max_epochs, min_epochs=min_epochs, max_steps=max_steps, min_steps=min_steps, max_time=max_time, limit_train_batches=limit_train_batches, limit_val_batches=limit_val_batches, limit_test_batches=limit_test_batches, limit_predict_batches=limit_predict_batches, overfit_batches=overfit_batches, val_check_interval=val_check_interval, check_val_every_n_epoch=check_val_every_n_epoch, num_sanity_val_steps=num_sanity_val_steps, log_every_n_steps=log_every_n_steps, enable_checkpointing=enable_checkpointing, enable_progress_bar=enable_progress_bar, enable_model_summary=enable_model_summary, accumulate_grad_batches=accumulate_grad_batches, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, deterministic=deterministic, benchmark=benchmark, inference_mode=inference_mode, use_distributed_sampler=use_distributed_sampler, profiler=profiler, detect_anomaly=detect_anomaly, barebones=barebones, plugins=plugins, sync_batchnorm=sync_batchnorm, reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs, default_root_dir=default_root_dir)

    
    def fit(self, model, train_dataloaders = None, val_dataloaders = None, datamodule = None, ckpt_path = None):
        # Create an empty DataModule to make the Lightning process run smoothly
        if train_dataloaders is None :
            train_dataloaders = EmptyDataModule()

        return super().fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)