import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pytorch_lightning as pl
from math import floor
import sys
sys.path.append(".")
from modules.data_preparation.process_data import FreiHAND  # show_dataset_sample
from modules.architectures.unet import UNet
from modules.pipelines.unet_pipeline import UnetPipeline
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

# General configurations for the training
config = {
    "data_dir": str(Path.cwd().joinpath("database/FreiHAND_pub_v2")),
    "model_name": "hand_pose",
    "epochs": 30,
    "batch_size": 48,
    "learning_rate": 0.1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "early_stop": True
}

# Callbacks
early_stop_callback = EarlyStopping(monitor="val_loss_epoch", min_delta=0.00, patience=10, verbose=False, mode="min")
checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch", filename="hand_pose_{epoch:d}_{val_loss:.2f}")

if __name__ == '__main__':
    # Generate the dataloaders
    train_dataset = FreiHAND(device=config["device"], data_dir=config["data_dir"], set_type="train")
    val_dataset = FreiHAND(device=config["device"], data_dir=config["data_dir"], set_type="val")
    train_dataloader = DataLoader(train_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, config["batch_size"], shuffle=False, drop_last=True, num_workers=2)

    print("--- Data information ---")
    print(f'Frames to train: {len(train_dataloader.dataset)} = {floor(len(train_dataloader.dataset) / config["batch_size"])} batches')
    print(f'Frames to validate: {len(val_dataloader.dataset)} = {floor(len(val_dataloader.dataset) / config["batch_size"])} batches')
    print(f'Early stop: {config["early_stop"]}\n')
    # show_dataset_sample(train_dataset, n_samples=8)

    # Create the pipeline for this model
    model = UnetPipeline(UNet(), config=config)

    # Training configuration
    use_gpu = 1 if config["device"] == torch.device("cuda") else 0
    callbacks_to_use = [checkpoint_callback]
    if config["early_stop"]:
        callbacks_to_use.append(early_stop_callback)

    trainer = pl.Trainer(gpus=use_gpu, num_nodes=0, precision=16, max_epochs=config["epochs"], callbacks=callbacks_to_use)
    print(f'Path to logs and saved models: {trainer.log_dir}\n')

    # The learning process
    trainer.fit(model, val_dataloader, val_dataloader)
