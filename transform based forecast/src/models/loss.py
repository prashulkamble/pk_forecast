import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class PoissonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y):
        return torch.sum(y * yhat - torch.exp(yhat))


class TweedieLoss(nn.Module):
    """
    Custom loss fuction designed to minimize the deviance using stochastic gradient descent
    tweedie deviance from McCullagh 1983

    """

    def __init__(self):
        super().__init__()

    def forward(self, predicted, observed):
        d = -2 * self._QLL(predicted, observed)
        #     loss = (weight*d)/1

        return torch.mean(d)

    def _QLL(self, predicted, observed):
        p = torch.tensor(1.5)
        QLL = QLL = torch.pow(predicted, (-p)) * (
            ((predicted * observed) / (1 - p)) - ((torch.pow(predicted, 2)) / (2 - p))
        )

        return QLL
