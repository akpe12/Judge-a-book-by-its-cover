import pytorch_lightning as pl

from util.config import ex
from util.dataset.SourceRetrievalDataModule import SourceRetrievalDataModule
from util.model.SourceRetrievalModule import SourceRetrievalModule
import copy
import os
# from lightning.pytorch.strategies.ddp import DDPStrategy

import warnings
warnings.filterwarnings(action='ignore')

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    
    # Print config
    for key, val in _config.items():
        key_str = "{}".format(key) + (" " * (30 - len(key)))
        print(f"{key_str}   =   {val}")    
    
    pl.seed_everything(_config["seed"])   
    
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # save_top_k=3,
        verbose=True,
        monitor="val/acc",
        filename='epoch={epoch}-step={step}-val_acc={val/acc:.5f}',
        mode="max",
        save_last=True,
        auto_insert_metric_name=False
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )


    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    
    class DatasetCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, model):
            # print('before_set_phase')
            trainer.datamodule.traindataset_reinit_at_train_epoch_end(trainer.current_epoch + 1)    
            
    dataset_callback = DatasetCallback()
    
    callbacks = [checkpoint_callback, lr_callback, dataset_callback]

    accumulate_grad_batches = max(_config["batch_size"] // (
        _config["per_gpu_batch_size"] * len(_config['gpus']) * _config["num_nodes"]
    ), 1)

    dm = SourceRetrievalDataModule(_config=_config)
    if _config['mode'] == 'test':
        model = SourceRetrievalModule.load_from_checkpoint(_config["load_path"], _config=_config, num_labels=_config['num_labels'])
    else:
        model = SourceRetrievalModule(_config=_config)

    trainer = pl.Trainer(
        gpus=_config['gpus'],
        max_steps=_config["max_steps"],
        accelerator="ddp",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=_config['val_check_interval'],
        gradient_clip_val=1.0,
        reload_dataloaders_every_epoch=True
        )

    
    if _config['mode'] == 'test':
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)