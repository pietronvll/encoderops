from typing import List, Union

import lightning
import torch
from mlcolvar.core import BaseGNN, FeedForward, Normalization
from mlcolvar.core.loss import ReduceEigenvaluesLoss
from mlcolvar.core.stats import TICA
from mlcolvar.cvs import BaseCV

__all__ = ["DeepTICA"]


class _RegSpectralLoss(torch.nn.Module):
    def __init__(self, reg: float = 1e-5):
        super().__init__()
        self.reg = reg

    def regularization_term(self, inputs, lagged):
        inputs_norm2 = (torch.linalg.matrix_norm(inputs)) ** 2
        lagged_norm2 = (torch.linalg.matrix_norm(lagged)) ** 2
        return self.reg * (inputs_norm2 + lagged_norm2) / 2

    def forward(self, inputs, lagged):
        return self.__call__(inputs, lagged)

    def noreg(self, inputs, lagged):
        return F.l2_contrastive_loss(inputs, lagged)

    def __call__(self, inputs, lagged):
        loss = F.l2_contrastive_loss(inputs, lagged)
        reg = self.regularization_term(inputs, lagged)
        return loss + reg

class RegSpectralLoss(BaseCV, lightning.LightningModule):


    DEFAULT_BLOCKS = ["norm_in", "nn", "tica"]
    MODEL_BLOCKS = ["nn", "tica"]

    def __init__(self, 
                 model: Union[List[int], FeedForward, BaseGNN], 
                 n_cvs: int = None, 
                 options: dict = None, **kwargs):
        """
        Define a Deep-TICA CV, composed of a neural network module and a TICA object.
        By default a module standardizing the inputs is also used.

        Parameters
        ----------
        model : list or FeedForward or BaseGNN
            Determines the underlying machine-learning model. One can pass:
            1. A list of integers corresponding to the number of neurons per layer of a feed-forward NN.
               The model Will be automatically intialized using a `mlcolvar.core.nn.feedforward.FeedForward` object.
               The CV class will be initialized according to the DEFAULT_BLOCKS.
            2. An externally intialized model (either `mlcolvar.core.nn.feedforward.FeedForward` or `mlcolvar.core.nn.graph.BaseGNN` object).
               The CV class will be initialized according to the MODEL_BLOCKS.
        n_cvs : int, optional
            Number of cvs to optimize, default None (= last layer)
        options : dict[str, Any], optional
            Options for the building blocks of the model, by default {}.
            Available blocks: ['norm_in','nn','tica'].
            Set 'block_name' = None or False to turn off that block
        """
        super().__init__(model, **kwargs)

        # =======   LOSS  =======
        # Maximize the squared sum of all the TICA eigenvalues.
        self.loss_fn = ReduceEigenvaluesLoss(mode="sum2")
        # here we need to override the self.out_features attribute
        self.out_features = n_cvs

        # ======= OPTIONS =======
        # parse and sanitize
        options = self.parse_options(options)

        # ======= BLOCKS =======

        if not self._override_model:
            # initialize norm_in
            o = "norm_in"
            if (options[o] is not False) and (options[o] is not None):
                self.norm_in = Normalization(self.in_features, **options[o])

            # initialize nn
            o = "nn"
            self.nn = FeedForward(self.layers, **options[o])
        
        elif self._override_model:
            self.nn = model

        # initialize tica
        o = "tica"
        self.tica = TICA(self.nn.out_features, n_cvs, **options[o])

    def forward_nn(self, x: torch.Tensor) -> torch.Tensor:
        if not self._override_model:
            if self.norm_in is not None:
                x = self.norm_in(x)
        x = self.nn(x)
        return x

    def set_regularization(self, c0_reg=1e-6):
        """
        Add identity matrix multiplied by `c0_reg` to correlation matrix C(0) to avoid instabilities in performin Cholesky and .

        Parameters
        ----------
        c0_reg : float
            Regularization value for C_0.
        """
        self.tica.reg_C_0 = c0_reg

    def training_step(self, train_batch, batch_idx):
        """Compute and return the training loss and record metrics.
        1) Calculate the NN output
        2) Remove average (inside forward_nn)
        3) Compute TICA
        """
        # =================get data===================
        if isinstance(self.nn, FeedForward):
            x_t = train_batch["data"]
            x_lag = train_batch["data_lag"]
            w_t = train_batch["weights"]
            w_lag = train_batch["weights_lag"]
        elif isinstance(self.nn, BaseGNN):
            x_t = self._setup_graph_data(train_batch, key='data_list')
            x_lag = self._setup_graph_data(train_batch, key='data_list_lag')
            w_t = x_t['weight']
            w_lag = x_lag['weight']

        # =================forward====================
        f_t = self.forward_nn(x_t)
        f_lag = self.forward_nn(x_lag)
        # ===================tica=====================
        eigvals, _ = self.tica.compute(
            data=[f_t, f_lag], weights=[w_t, w_lag], save_params=True
        )
        # ===================loss=====================
        loss = self.loss_fn(eigvals)
        # ====================log=====================
        name = "train" if self.training else "valid"
        loss_dict = {f"{name}_loss": loss}
        eig_dict = {f"{name}_eigval_{i+1}": eigvals[i] for i in range(len(eigvals))}
        self.log_dict(dict(loss_dict, **eig_dict), on_step=True, on_epoch=True)
        return loss


def test_deep_tica():
    # tests
    import numpy as np
    from mlcolvar.data import DictModule
    from mlcolvar.utils.timelagged import create_timelagged_dataset

    # create dataset
    # X = np.loadtxt("mlcolvar/tests/data/mb-mcmc.dat")
    X = torch.randn((10000, 2))
    dataset = create_timelagged_dataset(X, lag_time=1)
    datamodule = DictModule(dataset, batch_size=10000)

    # create cv
    print()
    print('NORMAL')
    print()
    layers = [2, 10, 10, 2]
    model = DeepTICA(layers, n_cvs=1)

    # change loss options
    model.loss_fn.mode = "sum2"

    # create trainer and fit
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)

    model.eval()
    with torch.no_grad():
        s = model(X).numpy()
    print(X.shape, "-->", s.shape)


    print()
    print('EXTERNAL')
    print()
    ff_model = FeedForward(layers=layers)
    model = DeepTICA(ff_model, n_cvs=1)

    # change loss options
    model.loss_fn.mode = "sum2"

    # create trainer and fit
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=None, enable_checkpointing=False
    )
    trainer.fit(model, datamodule)

    model.eval()
    with torch.no_grad():
        s = model(X).numpy()
    print(X.shape, "-->", s.shape)

    
    # gnn external
    print()
    print('GNN')
    print()
    from mlcolvar.core.nn.graph.schnet import SchNetModel
    from mlcolvar.data.graph.utils import create_test_graph_input
    gnn_model = SchNetModel(2, 0.1, [1, 8])
    model = DeepTICA(gnn_model, n_cvs=1)

    # change loss options
    model.loss_fn.mode = "sum2"

    # create trainer and fit
    trainer = lightning.Trainer(
        max_epochs=1, log_every_n_steps=2, logger=False, enable_checkpointing=False, enable_model_summary=False,
    )

    dataset = create_test_graph_input(output_type='dataset', n_samples=200, n_states=2)
    lagged_dataset = create_timelagged_dataset(dataset, logweights=torch.randn(len(dataset)))
    
    datamodule = DictModule(dataset=lagged_dataset)
    trainer.fit(model, datamodule)

    model.eval()
    with torch.no_grad():
        example_input_graph_test = create_test_graph_input(output_type='example', n_atoms=4, n_samples=3, n_states=2)
        s = model(example_input_graph_test).numpy()
    print(X.shape, "-->", s.shape)

if __name__ == "__main__":
    test_deep_tica()