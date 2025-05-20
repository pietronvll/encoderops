import torch
import torch.distributed
from mlcolvar.core.nn.graph.schnet import SchNetModel

class SchNet(torch.nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model = SchNetModel(**model_args)
    
    def forward(self, data, lagged: bool = False):
        # data
        if lagged:
            data = self._setup_graph_data(data, key="item_lag")
        else:
            data = self._setup_graph_data(data)
        return self.model(data)

    @staticmethod
    def _setup_graph_data(train_batch, key: str = "item"):
        data = train_batch[key]
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        return data
    
    


class EMACovariance(torch.nn.Module):
    def __init__(self, feature_dim: int, momentum: float = 0.01, center: bool = True):
        super().__init__()
        self.is_centered = center
        self.momentum = momentum
        self.register_buffer("mean_X", torch.zeros(feature_dim))
        self.register_buffer("cov_X", torch.eye(feature_dim))
        self.register_buffer("mean_Y", torch.zeros(feature_dim))
        self.register_buffer("cov_Y", torch.eye(feature_dim))
        self.register_buffer("cov_XY", torch.eye(feature_dim))
        self._has_been_called_once = False
    
    @torch.no_grad()
    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.ndim == 2
        assert X.shape == Y.shape
        assert X.shape[1] == self.mean_X.shape[0]
        if not self._has_been_called_once:
            self.first_forward(X, Y)
        else:
            mean_X = X.mean(dim=0, keepdim=True)
            mean_Y = Y.mean(dim=0, keepdim=True)

            self.mean_X = self._inplace_EMA(mean_X[0], self.mean_X)
            self.mean_Y = self._inplace_EMA(mean_Y[0], self.mean_Y)

            if self.is_centered:
                X = X - self.mean_X
                Y = Y - self.mean_Y

            cov_X = torch.mm(X.T, X) / X.shape[0]
            cov_Y = torch.mm(Y.T, Y) / Y.shape[0]
            cov_XY = torch.mm(X.T, Y) / X.shape[0]
            self.cov_X = self._inplace_EMA(cov_X, self.cov_X)
            self.cov_Y = self._inplace_EMA(cov_Y, self.cov_Y)
            self.cov_XY = self._inplace_EMA(cov_XY, self.cov_XY)

    def _first_forward(self, X: torch.Tensor, Y: torch.Tensor):
        mean_X = X.mean(dim=0, keepdim=True)
        self._inplace_set(mean_X[0], self.mean_X)
        mean_Y = Y.mean(dim=0, keepdim=True)
        self._inplace_set(mean_Y[0], self.mean_Y)
        if self.is_centered:
            X = X - self.mean_X
            Y = Y - self.mean_Y

        cov_X = torch.mm(X.T, X) / X.shape[0]
        cov_Y = torch.mm(Y.T, Y) / Y.shape[0]
        cov_XY = torch.mm(X.T, Y) / X.shape[0]
        self._inplace_set(cov_X, self.cov_X)
        self._inplace_set(cov_Y, self.cov_Y)
        self._inplace_set(cov_XY, self.cov_XY)
        self._has_been_called_once = True

    def _inplace_set(self, update, current):
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(update, op=torch.distributed.ReduceOp.SUM)
            update /= torch.distributed.get_world_size()
        current.copy_(update)

    def _inplace_EMA(self, update, current):
        alpha = 1 - self.momentum
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(update, op=torch.distributed.ReduceOp.SUM)
            update /= torch.distributed.get_world_size()

        current.mul_(alpha).add_(update, alpha=self.momentum)


class EuclideanNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor):
        return torch.nn.functional.normalize(X, dim=-1)
        