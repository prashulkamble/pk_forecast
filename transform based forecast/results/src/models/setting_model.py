import math

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import (
    _LRScheduler,
    ReduceLROnPlateau,
    StepLR,
    CosineAnnealingLR,
)
from ..utils.utils import stop_watch
from ..models.architecture import TransformerModel
from ..models.loss import RMSELoss


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class AdaBound(Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        final_lr=0.1,
        gamma=1e-3,
        eps=1e-8,
        weight_decay=0,
        amsbound=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay,
            amsbound=amsbound,
        )
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group["lr"], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsbound", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsbound = group["amsbound"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsbound:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group["final_lr"] * group["lr"] / base_lr
                lower_bound = final_lr * (1 - 1 / (group["gamma"] * state["step"] + 1))
                upper_bound = final_lr * (1 + 1 / (group["gamma"] * state["step"]))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss


class SamplingRateScheduler(object):
    def __init__(self, decay_schedules="inverse_sigmoid_decay", *args, **kwargs):
        if decay_schedules == "linear_decay":
            self.step_sampling_rate = self.linear_decay(*args, **kwargs)
        if decay_schedules == "exponential_decay":
            self.step_sampling_rate = self.exponential_decay(*args, **kwargs)
        if decay_schedules == "inverse_sigmoid_decay":
            self.step_sampling_rate = self.inverse_sigmoid_decay(*args, **kwargs)

        self.epoch = 0
        self.sampling_rate = self.step_sampling_rate(0)

    def step(self):
        self.epoch += 1
        sampling_rate = self.step_sampling_rate(self.epoch)
        self.sampling_rate = sampling_rate

    def linear_decay(self, k, c, eps=0.01):
        def linear_decay_(epoch):
            gt_sampling_rate = max([eps, k - c * epoch])
            return gt_sampling_rate

        return linear_decay_

    def exponential_decay(self, k):
        def exponential_decay_(epoch):
            gt_sampling_rate = k ** epoch
            return gt_sampling_rate

        return exponential_decay_

    def inverse_sigmoid_decay(self, k, start, end, slope):
        def inverse_sigmoid_decay_(epoch):
            gt_sampling_rate = (
                (start - end) / (1 + np.exp(epoch - (k / 2)) ** slope)
            ) + end
            return gt_sampling_rate

        return inverse_sigmoid_decay_


def setting_model(
    X,
    cat_emb,
    all_cat_cols,
    all_num_cols,
    model_params,
    opt_params,
    lr_params,
    warmup_params,
    sr_params,
):
    cat_emb_map = {col: (X[col].nunique(), cat_emb[col]) for col in all_cat_cols}

    if (
        sum([v[0] if v[1] == 0 else v[1] for v in cat_emb_map.values()])
        + len(all_num_cols)
    ) % 2 == 1:
        cat_emb_map[all_cat_cols[0]] = (
            X[all_cat_cols[0]].nunique(),
            cat_emb[all_cat_cols[0]] + 1,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(cat_embs=cat_emb_map, **model_params, device=device).to(
        device
    )

    criterion = RMSELoss().to(device)
    # criterion = nn.MSELoss().to(device)
    # criterion = nn.PoissonNLLLoss(log_input=True, full=True, reduction='mean').to(device)
    # criterion = TweedieLoss()

    # optimizer = AdaBound(model.parameters(), **opt_params)
    optimizer = Adam(model.parameters(), **opt_params)

    scheduler_base = CosineAnnealingLR(optimizer, **lr_params)
    lr_scheduler = GradualWarmupScheduler(
        optimizer, after_scheduler=scheduler_base, **warmup_params
    )

    sr_scheduler = SamplingRateScheduler(**sr_params)

    return model, optimizer, criterion, lr_scheduler, sr_scheduler
