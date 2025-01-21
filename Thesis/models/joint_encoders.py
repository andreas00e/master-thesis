import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F


class JointDescriptioEncoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, dynamic_joint_description, dynamic_joint_state, stability_epsilon):
        super().__init__()
        
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 



        self.dynamic_joint_description = dynamic_joint_description 
        self.dynamic_joint_state = dynamic_joint_state

        self.joint_descritption_encoder = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), 
                                           nn.LayerNorm(), 
                                           nn.ELU(),
                                           nn.Linear(self.hidden_dim, self.output_dim),
                                           nn.Tanh(), 
                                           nn.Softmax())

        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)