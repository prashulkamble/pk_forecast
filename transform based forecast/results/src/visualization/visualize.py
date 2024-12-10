import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from ..models.eval_model import root_mean_squared_error


def plot_sales(X_sum, X_sales, d_cols):
    sorted_d_cols = [col for col in d_cols if col in X_sum.index]

    fig, ax = plt.subplots(1, 1, figsize=[6.4 * 1.5, 4.8])
    ax.plot(X_sum[sorted_d_cols])
    ax.set_xticks(sorted_d_cols[::28])
    ax.tick_params(axis="x", labelrotation=45)
    fig.show()

    fig, ax = plt.subplots(1, 1)
    X_sales.hist(range=(X_sales.quantile(0.01), X_sales.quantile(0.99)), ax=ax)
    fig.show()


def plot_lr_and_sr(epochs, optimizer, lr_scheduler, sr_scheduler):
    lr_logs, sr_logs = [], []
    for _ in range(epochs):
        lr_logs.append(optimizer.param_groups[0]["lr"])
        sr_logs.append(sr_scheduler.sampling_rate)
        sr_scheduler.step()
        lr_scheduler.step()

    fig, ax = plt.subplots(1, 2, figsize=[6.4 * 2, 4.8])
    ax[0].set_title("Learning_rate")
    ax[0].plot(lr_logs)
    ax[1].set_title("Sampling_rate")
    ax[1].plot(sr_logs)
    fig.show()


def plot_losses(model, losses):
    fig, ax = plt.subplots(1, 1, figsize=[6.4, 4.8])
    for k, v in losses.items():
        ax.plot(v, label=k)
    ax.legend()
    ax.set_title(model.criterion._get_name())
    fig.show()


def plot_eval(fpath, output, ground_truth):
    Path(fpath).parent.mkdir(parents=True, exist_ok=True)
    loss = root_mean_squared_error(output.values, ground_truth.values)

    fig, ax = plt.subplots(1, 1)
    ax.plot(output.sum(0), label="Forecast")
    ax.plot(ground_truth.sum(0), label="Actual")
    ax.set_title("RMSE:{:.5f}".format(loss))
    ax.set_xticks(ground_truth.columns[::2])
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend()
    fig.show()
    fig.savefig(fpath)


def plot_eval_per_group(save_dir, output, ground_truth, group_col, d_cols):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for group_id, X_group in ground_truth.groupby(group_col):
        fig, ax = plt.subplots(1, 1)
        loss = root_mean_squared_error(
            output.loc[X_group.index, d_cols].values, X_group[d_cols].values
        )
        ax.plot(output.loc[X_group.index, d_cols].sum(0), label="Forecast")
        ax.plot(X_group[d_cols].sum(0), label="Actual")
        ax.set_title(f"{group_col}_{group_id}" + "(RMSE={:.5f})".format(loss))
        ax.set_xticks(d_cols[::2])
        ax.tick_params(axis="x", labelrotation=45)
        ax.legend()
        fig.show()
        fig.savefig(Path(save_dir, f"{group_col}_{group_id}"))


def plot_prediction(fpath, sales_train, output):
    zeros = np.zeros((output.shape[-1]))
    zeros[:] = np.nan
    true_train_fixed = np.concatenate([sales_train.sum(0), zeros])

    zeros = np.zeros((true_train_fixed.shape[0] - output.shape[-1]))
    zeros[:] = np.nan
    output_test_fixed = np.concatenate([zeros, output.sum(0)])

    true_train_fixed[-output.shape[-1]] = output_test_fixed[-output.shape[-1]]

    d_cols = list(sales_train.columns) + list(output.columns)
    fig, ax = plt.subplots(1, 1, figsize=[6.4 * 1.5, 4.8])
    ax.plot(d_cols, true_train_fixed, label="train(Actual)")
    ax.plot(d_cols, output_test_fixed, label="public(Forecast)")
    ax.set_xticks(d_cols[::28])
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend()
    fig.show()
    fig.savefig(fpath)
