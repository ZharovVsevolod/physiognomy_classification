import hydra
from hydra.core.config_store import ConfigStore

from physiognomy_classification.config import Params, Scheduler_OneCycleLR, Scheduler_ReduceOnPlateau
from physiognomy_classification.models.shell import Model_Lightning_Shell
from physiognomy_classification.data import ChalearnDataModule

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import os
from dotenv import load_dotenv

load_dotenv()

cs = ConfigStore.instance()
cs.store(name="params", node=Params)
cs.store(group="scheduler", name="base_rop", node=Scheduler_ReduceOnPlateau)
cs.store(group="scheduler", name="base_oclr", node=Scheduler_OneCycleLR)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    L.seed_everything(cfg.training.seed)
    wandb.login(key=os.environ["WANDB_KEY"])

    dm = ChalearnDataModule(
        train_data_dir = cfg.data.train_data_dir,
        valid_data_dir = cfg.data.valid_data_dir,
        test_data_dir = cfg.data.test_data_dir,
        train_labels_pickle = cfg.data.train_labels_pickle,
        valid_labels_pickle = cfg.data.valid_labels_pickle,
        test_labels_pickle = cfg.data.test_labels_pickle,
        batch_size = cfg.training.batch
    )
    model = Model_Lightning_Shell(cfg)

    os.mkdir(cfg.training.wandb_path)
    wandb_log = WandbLogger(
        project = cfg.training.project_name, 
        name = cfg.training.train_name, 
        save_dir = cfg.training.wandb_path
    )

    checkpoint = ModelCheckpoint(
        dirpath = cfg.training.model_path,
        filename = "epoch_{epoch}-{val_loss:.2f}",
        save_top_k = cfg.training.save_best_of,
        monitor = cfg.training.checkpoint_monitor
    )
    lr_monitor = LearningRateMonitor(logging_interval = "epoch")
    early_stop = EarlyStopping(monitor = cfg.training.checkpoint_monitor, patience = cfg.training.early_stopping_patience)

    trainer = L.Trainer(
        max_epochs = cfg.training.epochs,
        accelerator = "auto",
        log_every_n_steps = 500,
        devices = 1,
        logger = wandb_log,
        callbacks = [checkpoint, lr_monitor, early_stop],
        fast_dev_run = 5
    )
    trainer.fit(model = model, datamodule = dm)

    wandb.finish()


if __name__ == "__main__":
    main()