import gc
import pandas as pd
import numpy as np
from ..utils.utils import stop_watch


def read_dataset(private_sub=False):
    if private_sub:
        sales_suffix = "evaluation"
    else:
        sales_suffix = "validation"

    try:
        sell_prices = pd.read_csv(
            "/kaggle/input/m5-forecasting-accuracy/sell_prices.csv"
        )
        sample_submission = pd.read_csv(
            "/kaggle/input/m5-forecasting-accuracy/sample_submission.csv",
            index_col="id",
        )
        calendar = pd.read_csv(
            "/kaggle/input/m5-forecasting-accuracy/calendar.csv", parse_dates=["date"]
        )
        sales_train = pd.read_csv(
            f"/kaggle/input/m5-forecasting-accuracy/sales_train_{sales_suffix}.csv"
        )
    except FileNotFoundError:
        sell_prices = pd.read_csv("m5-forecasting-accuracy/sell_prices.csv")
        sample_submission = pd.read_csv(
            "m5-forecasting-accuracy/sample_submission.csv", index_col="id",
        )
        calendar = pd.read_csv(
            "m5-forecasting-accuracy/calendar.csv", parse_dates=["date"]
        )
        sales_train = pd.read_csv(
            f"m5-forecasting-accuracy/sales_train_{sales_suffix}.csv"
        )
    return sell_prices, sample_submission, calendar, sales_train


def get_date_cols(
    sales_train, sample_submission, train_size, max_lag, private_sub=False
):
    level_cols = sales_train.columns[:6].tolist()
    d_cols = sales_train.columns[6:].tolist()
    validation_d_cols = [f"d_{d}" for d in range(1914, 1941 + 1)]
    evaluation_d_cols = [f"d_{d}" for d in range(1942, 1969 + 1)]
    train_d_cols = d_cols[-(train_size + max_lag) :]

    if private_sub:
        test_d_cols = evaluation_d_cols
    else:
        test_d_cols = validation_d_cols

    d2F_map = {col: f"F{i+1}" for i, col in enumerate(test_d_cols)}

    return level_cols, train_d_cols, test_d_cols, d2F_map


@stop_watch
def merge_dataset(
    sales_train,
    calendar,
    sell_prices,
    base_cols,
    level_cols,
    train_d_cols,
    test_d_cols,
    reduce_memory=True,
):

    sales_train[train_d_cols] = sales_train[train_d_cols].astype(np.float32)
    sales_train = sales_train[level_cols + train_d_cols]
    sales_train = sales_train.assign(**{col: np.nan for col in test_d_cols})
    gc.collect()

    X = pd.melt(
        sales_train,
        id_vars="id",
        value_vars=train_d_cols + test_d_cols,
        var_name="d",
        value_name="sales",
    )
    X = pd.merge(sales_train[level_cols], X, how="right", on="id")
    X = pd.merge(X, calendar, how="left", on="d")
    X = pd.merge(X, sell_prices, how="left", on=["store_id", "item_id", "wm_yr_wk"])

    X = X[level_cols + ["d", "sales", "sell_price"] + base_cols]

    if reduce_memory:
        del sales_train, calendar, sell_prices

    gc.collect()

    return X
