import gc
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    StandardScaler,
    PowerTransformer,
    QuantileTransformer,
)
import pandas as pd
import numpy as np

from ..utils.utils import stop_watch


class DataTransformer(object):
    def __init__(self, log_trans, std_trans, minmax_trans, season_diff_trans=0):
        self.apply_trans = False
        if log_trans:
            # self.power_trans = QuantileTransformer(output_distribution='normal', random_state=seed)
            # self.power_trans = PowerTransformer(method='yeo-johnson', standardize=False)
            # self.power_trans = PowerTransformer(method="box-cox", standardize=True)
            self.power_trans = PowerTransformerWrapper(
                method="box-cox", standardize=True
            )
            self.apply_trans = True
        else:
            self.power_trans = None

        self.season_diff_trans = 0
        if 0 < self.season_diff_trans:
            self.apply_trans = True

        if std_trans:
            self.apply_trans = True
            self.std_trans = StandardScaler()
        else:
            self.std_trans = None

        if minmax_trans:
            self.apply_trans = True
            self.minmax_trans = MinMaxScaler((-1.0, 1.0))
        else:
            self.minmax_trans = None

    def fit(self, X):
        if not self.apply_trans:
            return X

        X_ = X.copy()

        if self.power_trans is not None:
            self.power_trans.fit(X_ + 1)

        # if 0 < self.season_diff_trans:
        #     if self.power_trans is not None:
        #         X_ = self.power_trans.transform(X_)
        #     X_ = pd.DataFrame(X_).diff(self.season_diff_trans, axis=0).values

        if self.std_trans is not None:
            self.std_trans.fit(X_)

        if self.minmax_trans is not None:
            self.minmax_trans.fit(X_)

    def transform(self, X):
        if not self.apply_trans:
            return X

        X_ = X.copy()

        if self.power_trans is not None:
            X_ = self.power_trans.transform(X_ + 1)

        # if 0 < self.season_diff_trans:
        #     X_ = pd.DataFrame(X_).diff(self.season_diff_trans, axis=0).values

        if self.std_trans is not None:
            X_ = self.std_trans.transform(X_)

        if self.minmax_trans is not None:
            X_ = self.minmax_trans.transform(X_)

        return X_

    def inverse_transform(self, X, season_start_values=None):
        if not self.apply_trans:
            return X

        X_ = X.copy()

        if self.minmax_trans is not None:
            X_ = self.minmax_trans.inverse_transform(X_)

        if self.std_trans is not None:
            X_ = self.std_trans.inverse_transform(X_)

        # if season_start_values is not None:
        #     for i in range(self.season_diff_trans):
        #         X_[i :: self.season_diff_trans] = (
        #             np.cumsum(X_[i :: self.season_diff_trans], axis=0)
        #             + season_start_values[i]
        #         )

        if self.power_trans is not None:
            X_ = self.power_trans.inverse_transform(X_) - 1
            X_ = X_.clip(0.0)

        return X_


@stop_watch
def select_activate_items(X, train_d_cols, test_size, bptt_x, max_lag):
    # select activate sales
    required_d_cols = train_d_cols[-test_size * 2 - bptt_x - max_lag :]

    activate_keys = ["id", "date"]

    # select activate sales
    active_flag = X.groupby(["id"])["sales"].cumsum() != 0
    activated_item_id_d = (
        X.loc[active_flag, activate_keys]
        .drop_duplicates()
        .sort_values(activate_keys)
        .reset_index(drop=True)
    )
    activated_item_id_d["activate_flg"] = 1
    X = pd.merge(X, activated_item_id_d, on=activate_keys, how="left")

    X = X.loc[(X["d"].isin(required_d_cols)) | (X["activate_flg"] == 1), :]

    # complement sales of non-activate item_id
    id_dates = X.groupby("id")["d"].count()
    non_activate_id = id_dates[id_dates == len(required_d_cols) + test_size].index

    if len(non_activate_id) == 0:
        X.drop(["activate_flg"], axis=1, inplace=True)

        return X

    ## with nearest
    X_non_activate = X.loc[
        X["id"].isin(non_activate_id),
        ["id", "item_id", "date", "d", "sales", "activate_flg"],
    ]
    X_non_activate.loc[X_non_activate["activate_flg"].isnull(), "sales"] = np.nan
    X_non_activate = X_non_activate[
        (X_non_activate.groupby("id")["activate_flg"].cumsum() <= 7)
        | (X_non_activate["activate_flg"].isnull())
    ]
    future_w_cols = []
    for future_w in range(7, len(required_d_cols), 7):
        col = f"sales_futurew{future_w}"
        future_w_cols.append(col)
        X_non_activate[col] = X_non_activate.groupby("id")["sales"].shift(-future_w)

    idx = X_non_activate["sales"].isnull()
    X_non_activate.loc[idx, "sales"] = X_non_activate.loc[idx, future_w_cols].sum(1)
    X.loc[X_non_activate.index, "sales"] = X_non_activate["sales"]

    ## with rolling mean
    future_m_means = (
        X.loc[X["id"].isin(non_activate_id)]
        .sort_values(["id", "date"], ascending=False)
        .groupby(["id", "weekday"])["sales"]
        .rolling(4, min_periods=1)
        .mean()
    )
    future_m_means.index = future_m_means.index.get_level_values(2)

    future_w_means = (
        X.loc[X["id"].isin(non_activate_id)]
        .sort_values(["id", "date"], ascending=False)
        .groupby(["id"])["sales"]
        .rolling(7, min_periods=1)
        .mean()
    )
    future_w_means.index = future_w_means.index.get_level_values(1)

    future_means = (0.5 * future_m_means + 0.5 * future_w_means).sort_index()

    idx = X_non_activate[X_non_activate["activate_flg"].isnull()].index
    X.loc[idx, "sales"] = future_means[idx]

    X.drop(["activate_flg"], axis=1, inplace=True)

    return X


@stop_watch
def complement_missing(X):
    outlier_event_names = ["Christmas", "Thanksgiving", "NewYear"]
    X["sales_window_12d_mean"] = X.groupby(["id"])["sales"].apply(
        lambda x: x.rolling(window=13, center=True).mean().reset_index(drop=True)
    ).reset_index(drop=True)
    X["sales_window_6w_mean"] = X.groupby(["id", "weekday"])["sales"].apply(
    lambda x: x.rolling(window=7, center=True).mean().reset_index(drop=True)
    ).reset_index(drop=True)
    idx = (X["event_name_1"].isin(outlier_event_names)).values
    X.loc[idx, "sales"] = (
        0.5 * X.loc[idx, "sales_window_12d_mean"]
        + 0.5 * X.loc[idx, "sales_window_6w_mean"]
    )

    X.drop(["sales_window_12d_mean", "sales_window_6w_mean"], axis=1, inplace=True)

    return X


@stop_watch
def add_base_features(X):
    X["is_snap"] = (
        ((X["state_id"] == "CA") & (X["snap_CA"] == 1))
        | ((X["state_id"] == "TX") & (X["snap_TX"] == 1))
        | ((X["state_id"] == "WI") & (X["snap_WI"] == 1))
    ).astype(np.int8)

    # date
    X["quarter"] = X["date"].dt.quarter
    X["is_weekend"] = X["weekday"].isin(["Saturday", "Sunday"]).astype(np.int8)
    X["part_of_month"] = X["date"].apply(
        lambda x: "B" if x.day <= 10 else "M" if x.day <= 20 else "E"
    )
    return X


def feature_enginearing(
    X,
    train_d_cols,
    test_d_cols,
    lags,
    max_lag,
    sales_cat_cols,
    cat_cols,
    num_cols,
    diff_trans,
    dtrans_map={
        "sales": [False, True, False, 0],
        "sell_price": [False, True, False, 0],
    },
    clipping_range={"sales": (0.0, 1.0), "sell_price": (0.0, 1.0)},
    inplace=True,
):

    if inplace:
        X_ = X
    else:
        X_ = X.copy()

    sorted_d_cols = (
        X_[["date", "d"]].drop_duplicates().sort_values("date")["d"].tolist()
    )

    all_cat_cols = sales_cat_cols + cat_cols

    # fillna
    X_[all_cat_cols] = X_[all_cat_cols].fillna("NaN")
    X_[num_cols] = X_[num_cols].fillna(0.0).astype(np.float32)
    gc.collect()

    # categorical encoding
    print("categorical encoding...")
    order_cols = [col for col in all_cat_cols if col != "id"]
    le = OrdinalEncoder()
    le.fit(X_[order_cols])
    X_[order_cols] = le.transform(X_[order_cols]).astype(np.int16)

    id_enc = LabelEncoder()
    id_enc.fit(X_["id"])
    X_["id"] = id_enc.transform(X_["id"]).astype(np.int16)

    gc.collect()

    X_[all_cat_cols] = X_[all_cat_cols].astype(np.int16)
    gc.collect()

    # diff
    X["sell_price"] = X.groupby(["id"])["sell_price"].diff().fillna(0.0)
    if diff_trans:
        X["sales"] = X.groupby(["id"])["sales"].diff()
        first_rows = X.groupby(["id"])["date"].transform("first")

        X = X[X["date"] != first_rows]

    # clipping
    print("clipping...")
    for clipping_col, (clip_min, clip_max) in clipping_range.items():
        if clip_max < 1.0:
            X[f"{clipping_col}_max"] = X.groupby(["id"])[clipping_col].transform(
                lambda x: x.quantile(clip_max)
            )
            X[clipping_col] = X[[clipping_col, f"{clipping_col}_max"]].min(1)
            X.drop([f"{clipping_col}_max"], axis=1, inplace=True)

        if clip_min > 0.0:
            X[f"{clipping_col}_min"] = X.groupby(["id"])[clipping_col].transform(
                lambda x: x.quantile(clip_min)
            )
            X[clipping_col] = X[[clipping_col, f"{clipping_col}_min"]].max(1)
            X.drop([f"{clipping_col}_min"], axis=1, inplace=True)

    # scaling
    print("scaling...")
    list_dtrans = []
    for col, dtrains_params in dtrans_map.items():
        X_pivot = pd.pivot(X_[["id", "d", col]], index="id", columns="d", values=col)
        X_pivot = X_pivot.loc[:, sorted_d_cols]
        dtrans = DataTransformer(*dtrains_params)
        dtrans.fit(X_pivot.loc[:, train_d_cols].values.T)
        X_pivot.loc[:] = dtrans.transform(X_pivot.values.T).T
        X_pivot = pd.melt(
            X_pivot.reset_index(),
            id_vars="id",
            value_vars=X_pivot.columns,
            value_name=col,
        )
        X_[col] = (
            X_pivot.set_index(["id", "d"])
            .loc[X_.set_index(["id", "d"]).index, col]
            .values.astype(np.float32)
        )
        list_dtrans.append(dtrans)

    del X_pivot
    gc.collect()

    # lag_features
    print("lag features...")
    lags_cols = []
    for tau, period in lags:
        for step in range(tau, period + 1, tau):
            col = f"sales_shift{step}"
            X_[col] = X_.groupby("id")["sales"].shift(step).astype(np.float32)
            lags_cols.append(col)

    X_ = X_[(X_["d"].isin(test_d_cols)) | (~X_[f"sales_shift{max_lag}"].isnull())]

    # rolling window
    agg_cols = []
    X_agg = []
    X_agg_w = []

    # agg
    # print("aggrigate...")
    # agg_names = ['max', 'mean', 'std']
    # agg_names = ["min", "max", "mean", "std", "kurt", "skew"]

    # X_["sales_for_rolling_window"] = X_.groupby("id")["sales"].shift(len(test_d_cols))
    # idx = X_["sales_for_rolling_window"].isnull()
    # X_.loc[idx, "sales_for_rolling_window"] = X_.loc[idx, "sales"]

    # for period in [7, 28]:
    #     X_agg_ = (
    #         X_.groupby("id")["sales_for_rolling_window"]
    #         .rolling(period, min_periods=1)
    #         .agg(agg_names)
    #         .astype(np.float32)
    #     )
    #     X_agg_.columns = [
    #         f"sales_d_rolling{period}_{agg_name}" for agg_name in agg_names
    #     ]
    #     X_agg.append(X_agg_.sort_index(level=1))

    # for period in [12]:
    #     X_agg_ = (
    #         X_.groupby(["id", "weekday"])["sales_for_rolling_window"]
    #         .rolling(period)
    #         .agg(agg_names)
    #         .astype(np.float32)
    #     )
    #     X_agg_.columns = [
    #         f"sales_w_rolling{period}_{agg_name}" for agg_name in agg_names
    #     ]
    #     X_agg_w.append(X_agg_.sort_index(level=2))

    # X_['sales_w_diff'] = X_.groupby(['id', 'weekday'])['sales_for_rolling_window'].diff(1)
    # for period in [7]:
    #     X_agg_ = X_.groupby(['id'])['sales_w_diff'].rolling(period).agg(
    #         agg_names).astype(np.float32)
    #     X_agg_.columns = [f'sales_w_diff_d_rolling{period}_{agg_name}' for agg_name in agg_names]
    #     X_agg.append(X_agg_.sort_index(level=1))

    # X_['sales_diff'] = X_.groupby(['id'])['sales_for_rolling_window'].diff(1)
    # for period in [12]:
    #     X_agg_ = X_.groupby(['id', 'weekday'])['sales_diff'].rolling(period).agg(
    #         agg_names).astype(np.float32)
    #     X_agg_.columns = [f'sales_d_diff_w_rolling{period}_{agg_name}' for agg_name in agg_names]
    #     X_agg_w.append(X_agg_.sort_index(level=2))

    # X_agg = pd.concat(X_agg, axis=1)
    # X_agg = X_agg.fillna(method="bfill")

    # X_agg_w = pd.concat(X_agg_w, axis=1).sort_index(level=2)
    # X_agg_w = X_agg_w.fillna(method="bfill")

    # X_agg = pd.concat(
    #     [
    #         X_agg.sort_index(level=1).reset_index(drop=True),
    #         X_agg_w.sort_index(level=2).reset_index(drop=True),
    #     ],
    #     axis=1,
    # )

    # agg_cols = X_agg.columns.tolist()

    # X_ = pd.concat([X_.reset_index(drop=True), X_agg.reset_index(drop=True)], axis=1)
    # del (
    #     X_agg_,
    #     X_agg,
    #     X_agg_w,
    #     X_["sales_for_rolling_window"],
    #     # X_["sales_w_diff"],
    #     # X_["sales_diff"],
    # )

    # gc.collect()

    all_num_cols = lags_cols + agg_cols + num_cols

    X_ = X_[sales_cat_cols + ["d", "sales"] + cat_cols + all_num_cols]
    #     X = reduce_mem_usage(X)
    gc.collect()

    return X_, all_cat_cols, all_num_cols, list_dtrans, id_enc


class PowerTransformerWrapper(PowerTransformer):
    def __init__(self, method="yeo-johnson", standardize=True, copy=True):
        super(PowerTransformerWrapper, self).__init__(
            method=method, standardize=standardize, copy=copy
        )

    def fit(self, X, y=None):
        self = super().fit(X, y)
        self.lambdas_ = np.zeros_like(self.lambdas_)
        return self

    # def _fit(self, X, y=None, force_transform=False):
    #     X = self._check_input(X, check_positive=True, check_method=True)

    #     if not self.copy and not force_transform:  # if call from fit()
    #         X = X.copy()  # force copy so that fit does not change X inplace

    #     optim_function = {
    #         "box-cox": self._box_cox_optimize,
    #         "yeo-johnson": self._yeo_johnson_optimize,
    #     }[self.method]
    #     with np.errstate(invalid="ignore"):  # hide NaN warnings
    #         #             from IPython.core.debugger import Pdb; Pdb().set_trace()
    #         self.lambdas_ = np.array(
    #             [
    #                 optim_function(col)
    #                 if np.unique(col[~np.isnan(col)]).shape[0] != 1
    #                 else 0.0
    #                 for col in X.T
    #             ]
    #         )

    #     if self.standardize or force_transform:
    #         from scipy.special import boxcox

    #         transform_function = {
    #             "box-cox": boxcox,
    #             "yeo-johnson": self._yeo_johnson_transform,
    #         }[self.method]
    #         for i, lmbda in enumerate(self.lambdas_):
    #             with np.errstate(invalid="ignore"):  # hide NaN warnings
    #                 X[:, i] = transform_function(X[:, i], lmbda)

    #     if self.standardize:
    #         self._scaler = StandardScaler(copy=False)
    #         if force_transform:
    #             X = self._scaler.fit_transform(X)
    #         else:
    #             self._scaler.fit(X)

    #     return X

