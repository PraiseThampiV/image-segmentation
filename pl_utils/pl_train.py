import torch
import os, shutil
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

def train_model(lightn_model, checkpoint_path, del_pre_files=True, device=None, num_epochs=11): 
    """train using pytorch lightning routines

    Parameters
    ----------
    lightn_model : `torch.nn.model`
        model put in correct device
    checkpoint_path :str
        path of save  model
    del_pre_files : bool, optional
        whether to delete previously saved items, by default True
    """

    if del_pre_files:
        try:
            for filename in os.listdir(checkpoint_path):
                file_path = os.path.join(checkpoint_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))
        except Exception as e:
                print('Failed to delete contents from %s. Reason: %s' % (checkpoint_path, e))

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        filepath=checkpoint_path,
        mode="max"
    )

    num_gpus = 1 if device.type=='cuda' else None
    #val_check_interval should be an integer if using iterable dataset
    trainer = pl.Trainer(gpus=num_gpus, max_epochs=num_epochs, default_root_dir=checkpoint_path, 
    checkpoint_callback=checkpoint_callback, val_check_interval=1)
                        #,resume_from_checkpoint='lightning_logs/version_21/checkpoints/epoch=1.ckpt')    
    trainer.fit(lightn_model)
    return trainer
     # trainer.save_checkpoint(save_model_loc)
