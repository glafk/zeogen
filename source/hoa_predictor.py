import pytorch_lightning as pl
import hydra
import torch.nn.functional as F

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        # Here the args would be the config file
        self.save_hyperparameters()

    def configure_optimizers(self):
        print(self.hparams)
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
    

class HOAPredictor(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = hydra.utils.instantiate(self.hparams.gnn, _convert_="partial")

    def forward(self, batch):
        prediction = self.model(
            None,
            batch.frac_coords,
            batch.atom_types, 
            batch.num_atoms, 
            batch.lengths, 
            batch.angles, 
            batch.edge_index, 
            batch.to_jimages, 
            batch.num_bonds)

        return prediction

    def training_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = F.mse_loss(prediction.view(-1), batch.y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = F.mse_loss(prediction.view(-1), batch.y.float())
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        prediction = self(batch)
        loss = F.mse_loss(prediction.view(-1), batch.y.float())
        self.log('test_loss', loss)

