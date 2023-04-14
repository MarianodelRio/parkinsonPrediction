'''This file contains the model class.
The model architecture is defined as a LSTM layer and a linear layer.
'''


import pytorch_lightning as pl
import torch
import numpy as np

class LSTMmodel(pl.LightningModule):
    def __init__(self, hparams=None, num_ratios=4):
        super().__init__()
        self.save_hyperparameters(hparams)
        # Initialize model
        self.lstm = torch.nn.LSTM(input_size=7, hidden_size=128, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(128, num_ratios)

    def forward(self, x):
        x, _ = self.lstm(x) # Second element is the hidden state
        x = x[:,-1, :] # Take the last output
        x = self.linear(x)
        return x 

    def predict(self, x):
        self.eval()
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()
        with torch.no_grad():
            return self.forward(x)
        
    def training_step(self, batch, batch_idx):
        # Calculate training loss
        x, y = batch
        y_hat = self(x)
        if self.hparams.loss == 'mse':
            loss = torch.nn.functional.mse_loss(y_hat, y)
        else:
            raise ValueError('Loss function not recognized')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Calculate validation loss
        x, y = batch
        y_hat = self(x)
        if self.hparams.loss == 'mse':
            loss = torch.nn.functional.mse_loss(y_hat, y)
        else: 
            raise ValueError('Loss function not recognized')
        self.log('val_loss', loss)

    def configure_optimizers(self):
        # Return an optimizer
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), **self.hparams['optimizer_params'])
        return optimizer