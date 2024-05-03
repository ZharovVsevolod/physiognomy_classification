from typing import Any, Literal, List
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import lightning as L
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy

from physiognomy_classification.config import Params
from physiognomy_classification.models.model import ViT

class LabelsAccuracy:
    def __init__(self) -> None:
        pass

    def __call__(self, list_predictions, list_ground_truth) -> List[float]:
        if len(list_ground_truth) != len(list_predictions):
            print("Cannot compute label accuracy, list lengths do not match!")
            exit()
        num_predictions = len(list_predictions)
        one_minus_absolute_distances = [1 - abs(list_ground_truth[i] - list_predictions[i]) for i in range(num_predictions)]
        return sum(one_minus_absolute_distances) / num_predictions


class Model_Lightning_Shell(L.LightningModule):
    def __init__(self, args: Params) -> None:
        super().__init__()
        self.args = args
        
        if args.model.name == "vit":
            self.inner_model = ViT(
                img_size=args.model.image_size,
                patch_size=args.model.patch_size,
                in_chans=args.model.in_channels,
                num_classes=args.model.num_output_classes,
                embed_dim=args.model.embedding_dim,
                depth=args.model.layers,
                num_heads=args.model.heads,
                mlp_ratio=args.model.mlp_ratio,
                qkv_bias=args.model.qkv_bias,
                drop_rate=args.model.dropout,
                norm_type=args.model.norm_type
            )
        
        # self.metric = MulticlassAccuracy(num_classes=args.model.num_output_classes)
        self.metric = LabelsAccuracy()
        self.save_hyperparameters()

    def forward(self, x) -> Any:
        return self.inner_model(x)
    
    def loss(self, y, y_hat):
        return F.mse_loss(y, y_hat)
        # return F.cross_entropy(y, y_hat)

    def lr_scheduler(self, optimizer):
        if self.args.scheduler.name == "ReduceOnPlateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                patience=self.args.scheduler.patience, 
                factor=self.args.scheduler.factor
            )
        
        if self.args.scheduler.name == "OneCycleLR":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.args.training.lr * self.args.scheduler.expand_lr, 
                total_steps=self.args.training.epochs
            )

    def log_ocean_accuracy(self, out, y, name:Literal["train", "val", "test"]) -> None:
        ocean = self.metric(out, y)

        self.log((name + "_O"), ocean[0])
        self.log((name + "_C"), ocean[1])
        self.log((name + "_E"), ocean[2])
        self.log((name + "_A"), ocean[3])
        self.log((name + "_N"), ocean[4])

    def training_step(self, batch) -> STEP_OUTPUT:
        x, y = batch

        out = self(x)[:,-1,:]
        
        pred_loss = self.loss(out, y)

        self.log("train_loss", pred_loss)
        self.log_ocean_accuracy(out, y, "train")
        return pred_loss
    
    def validation_step(self, batch) -> STEP_OUTPUT:
        x, y = batch

        out = self(x)[:,-1,:]
        
        pred_loss = self.loss(out, y)
        self.log("val_loss", pred_loss)
        self.log_ocean_accuracy(out, y, "val")
    
    def test_step(self, batch) -> STEP_OUTPUT:
        x, y = batch

        out = self(x)[:,-1,:]
        
        pred_loss = self.loss(out, y)
        self.log("test_loss", pred_loss)
        self.log_ocean_accuracy(out, y, "test")
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.training.lr)
        sched = self.lr_scheduler(optimizer)
        if self.args.scheduler.name == "ReduceOnPlateau":
            return (
                {'optimizer': optimizer, 'lr_scheduler': {"scheduler": sched, "monitor": "val_loss"}},
            )
        if self.args.scheduler.name == "OneCycleLR":
            return(
                {'optimizer': optimizer, 'lr_scheduler': {"scheduler": sched}},
            )