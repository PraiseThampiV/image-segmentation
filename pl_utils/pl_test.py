import os
import torch
def test_model(model, checkpoint_path, trainer):
    """test model

    Parameters
    ----------
    model : `torch.nn.model`
        model
    checkpoint_path : str
        path
    trainer : `trainer`
        pytorch lightning trainer
    """
    #testing results after loading model from saved folder
    if trainer is None:
        return
    saved_model = model
    for filename in os.listdir(checkpoint_path):
        file_path = os.path.join(checkpoint_path, filename)
        if os.path.isfile(file_path) and "epoch" in file_path:
            save_model_path = file_path
            break
    checkpoint = torch.load(save_model_path)
    saved_model.load_state_dict(checkpoint['state_dict'])
    trainer.test(saved_model)