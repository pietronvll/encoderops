import timm
import torch
import torch.distributed
from linear_operator_learning.nn import MLP as lolMLP
import torch.nn as nn
import torch.nn.functional as F


class ResNet18(torch.nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.model = timm.create_model("resnet18", **model_args)

    def forward(self, data):
        return self.model(data)

    def prepare_batch(self, train_batch):
        x, y = train_batch["x"], train_batch["y"]
        return x, y


class MLP(torch.nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.model = lolMLP(**model_args)

    def forward(self, data):
        return self.model(data)

    def prepare_batch(self, train_batch):
        x, y = train_batch[0], train_batch[1]
        return x, y


class SchNet(torch.nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.model = SchNetModel(**model_args)

    def forward(self, data):
        return self.model(data)

    @staticmethod
    def _setup_graph_data(train_batch, key: str = "item"):
        data = train_batch[key]
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        return data

    def prepare_batch(self, train_batch):
        x_t = self._setup_graph_data(train_batch, key="item")
        x_t_lag = self._setup_graph_data(train_batch, key="item_lag")
        return x_t, x_t_lag


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
        self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))
        self._has_been_called_once = False

    @torch.no_grad()
    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        assert X.ndim == 2
        assert X.shape == Y.shape
        assert X.shape[1] == self.mean_X.shape[0]
        if not self.is_initialized.item():
            self._first_forward(X, Y)
        else:
            mean_X = X.mean(dim=0, keepdim=True)
            mean_Y = Y.mean(dim=0, keepdim=True)
            # Update means
            self._inplace_EMA(mean_X[0], self.mean_X)
            self._inplace_EMA(mean_Y[0], self.mean_Y)

            if self.is_centered:
                X = X - self.mean_X
                Y = Y - self.mean_Y

            cov_X = torch.mm(X.T, X) / X.shape[0]
            cov_Y = torch.mm(Y.T, Y) / Y.shape[0]
            cov_XY = torch.mm(X.T, Y) / X.shape[0]
            # Update covariances
            self._inplace_EMA(cov_X, self.cov_X)
            self._inplace_EMA(cov_Y, self.cov_Y)
            self._inplace_EMA(cov_XY, self.cov_XY)

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
        self.is_initialized = torch.tensor(True, dtype=torch.bool)

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
    


class MaskedCNN(nn.Module):
    def __init__(self,
                 in_chans,
                 num_classes,):
        super(MaskedCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_chans-1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Masked global pooling
        self.global_pool = MaskedGlobalPooling('avg')
        
        # Final embedding layer
        self.embedding = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x shape: [batch, 2, height, width] where channel 0=SST, channel 1=mask

        sst_data = x[:, :-1, :, :]  # SST channel
        mask = x[:, -1:, :, :]      # Mask channel
        
        # Convolutional layers with masking
        out = F.relu(self.bn1(self.conv1(sst_data)))
        out = self.pool(out)
        mask = F.max_pool2d(mask, 2, 2)  # Downsample mask
        
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        mask = F.max_pool2d(mask, 2, 2)
        
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        mask = F.max_pool2d(mask, 2, 2)
        
        out = F.relu(self.bn4(self.conv4(out)))
        # No pooling after last conv to preserve spatial resolution for masking
        
        # Masked global pooling - this is where land pixels are excluded
        embedding = self.global_pool(out, mask)  # [batch, 512]
        
        # Final embedding
        embedding = self.embedding(embedding)  # [batch, embedding_dim]
        
        return embedding
    
    def prepare_batch(self, batch):
        x, y = batch["x"], batch["y"]
        return x, y



class TinyMaskedCNN(torch.nn.Module):
    def __init__(self, in_chans, num_classes):
        super().__init__()
        
        # Reduced convolutional layers
        self.conv1 = torch.nn.Conv2d(in_chans - 1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, padding=1)
        
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.bn4 = torch.nn.BatchNorm2d(128)
        
        self.pool = torch.nn.MaxPool2d(2, 2)
        
        # Masked global pooling
        self.global_pool = MaskedGlobalPooling('avg')
        
        # Final embedding layer
        self.embedding = torch.nn.Linear(128, num_classes)
        
    def forward(self, x):
        sst_data = x[:, :-1, :, :]
        mask = x[:, -1:, :, :]

        out = torch.nn.functional.relu(self.bn1(self.conv1(sst_data)))
        out = self.pool(out)
        mask = torch.nn.functional.max_pool2d(mask, 2)

        out = torch.nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        mask = torch.nn.functional.max_pool2d(mask, 2)

        out = torch.nn.functional.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        mask = torch.nn.functional.max_pool2d(mask, 2)

        out = torch.nn.functional.relu(self.bn4(self.conv4(out)))
        # No pool after final conv

        pooled = self.global_pool(out, mask)  # [batch, 128]
        return self.embedding(pooled)         # [batch, num_classes]

    def prepare_batch(self, batch):
        return batch["x"], batch["y"]
    

class MaskedGlobalPooling(torch.nn.Module):
    def __init__(self, pool_type='avg'):
        super().__init__()
        self.pool_type = pool_type
    
    def forward(self, features, mask):
        """
        features: [batch, channels, height, width]
        mask: [batch, 1, height, width] - 1 for ocean, 0 for land
        """
        if self.pool_type == 'avg':
            # Masked average pooling
            masked_features = features * mask
            sum_features = torch.sum(masked_features, dim=[2, 3])  # [batch, channels]
            valid_pixels = torch.sum(mask, dim=[2, 3])  # [batch, 1]
            
            # Avoid division by zero
            valid_pixels = torch.clamp(valid_pixels, min=1e-8)
            return sum_features / valid_pixels
            
        elif self.pool_type == 'max':
            # Masked max pooling - set land pixels to very negative values
            masked_features = torch.where(
                mask.expand_as(features) == 1,
                features,
                torch.full_like(features, -1e9)
            )
            return torch.nn.functional.adaptive_max_pool2d(masked_features, 1).squeeze(-1).squeeze(-1)
