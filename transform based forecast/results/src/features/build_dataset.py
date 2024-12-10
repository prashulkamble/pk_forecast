import gc
import torch
import numpy as np
from ..utils.utils import stop_watch


class TSdatasets(torch.utils.data.Dataset):
    def __init__(
        self,
        X,
        split_d_cols,
        all_num_cols,
        all_cat_cols,
        bptt_x,
        bptt_y,
        max_lag,
        mask=False,
    ):
        id_col = all_cat_cols[0]

        if not mask:
            (
                self.en_i,
                self.en_cat_i,
                self.de_i,
                self.de_cat_i,
                self.de_o,
            ) = self.create_dataset(
                X,
                split_d_cols,
                all_num_cols,
                all_cat_cols,
                id_col,
                bptt_x,
                bptt_y,
                max_lag,
            )
            self.de_o = self.de_o.astype(np.float32)
        else:
            (
                self.en_i,
                self.en_cat_i,
                self.de_i,
                self.de_cat_i,
            ) = self.create_dataset_mask(
                X,
                split_d_cols,
                all_num_cols,
                all_cat_cols,
                id_col,
                bptt_x,
                bptt_y,
                max_lag,
            )
            self.de_i = np.nan_to_num(self.de_i)

        self.en_i = self.en_i.astype(np.float32)
        self.en_cat_i = self.en_cat_i.astype(int)
        self.de_i = self.de_i.astype(np.float32)
        self.de_cat_i = self.de_cat_i.astype(int)
        self.mask = mask

    def __len__(self):
        return len(self.en_i)

    def __getitem__(self, idx):
        en_input = self.en_i[idx]
        en_cat_input = self.en_cat_i[idx]
        de_input = self.de_i[idx]
        de_cat_input = self.de_cat_i[idx]

        if self.mask:
            return (en_input, en_cat_input, de_input, de_cat_input)
        else:
            de_output = self.de_o[idx]
            return (en_input, en_cat_input, de_input, de_cat_input, de_output)

    def create_dataset(
        self,
        X,
        split_d_cols,
        all_num_cols,
        all_cat_cols,
        id_col,
        bptt_x,
        bptt_y,
        max_lag,
    ):
        en_i, de_i, en_cat_i, de_cat_i, de_o = [], [], [], [], []

        X_split = X[X["d"].isin(split_d_cols)]

        for train_i in range(
            ((len(split_d_cols) - max_lag) % bptt_y) + 1,
            len(split_d_cols) - bptt_x - bptt_y + 2,
            bptt_y,
        ):
            en_i_d_cols_batch = split_d_cols[train_i : train_i + bptt_x]
            de_i_d_cols_batch = split_d_cols[
                train_i + bptt_x - 1 : train_i + bptt_x - 1 + bptt_y
            ]
            #         print(en_i_d_cols_batch[0], en_i_d_cols_batch[-1], len(en_i_d_cols_batch),
            #               de_i_d_cols_batch[0], de_i_d_cols_batch[-1], len(de_i_d_cols_batch))

            # encode_decode numeric input
            X_split_batch = X_split.loc[
                X_split["d"].isin(en_i_d_cols_batch + de_i_d_cols_batch[1:]), :
            ]
            id_cols_batch = X_split_batch.groupby([id_col])["d"].count() == len(
                en_i_d_cols_batch + de_i_d_cols_batch[1:]
            )
            X_split_batch = X_split_batch.loc[
                X_split_batch[id_col].isin(id_cols_batch[id_cols_batch].index),
                all_num_cols,
            ]

            X_split_batch = X_split_batch.values.reshape(
                -1, bptt_x + bptt_y - 1, len(all_num_cols)
            )

            en_i.append(X_split_batch[:, : len(en_i_d_cols_batch)])
            de_i.append(X_split_batch[:, len(en_i_d_cols_batch) - 1 :])

            # encode_decode categorical input
            X_split_batch = X_split.loc[
                X_split["d"].isin(en_i_d_cols_batch + de_i_d_cols_batch[1:]), :
            ]
            X_split_batch = X_split_batch.loc[
                X_split_batch[id_col].isin(id_cols_batch[id_cols_batch].index),
                all_cat_cols,
            ]
            X_split_batch = X_split_batch.values.reshape(
                -1, bptt_x + bptt_y - 1, len(all_cat_cols)
            )

            en_cat_i.append(X_split_batch[:, : len(en_i_d_cols_batch)])
            de_cat_i.append(X_split_batch[:, len(en_i_d_cols_batch) - 1 :])

            # output
            X_split_batch = X_split.loc[X_split["d"].isin(de_i_d_cols_batch), :]
            X_split_batch = X_split_batch.loc[
                X_split_batch[id_col].isin(id_cols_batch[id_cols_batch].index), "sales"
            ]

            de_o.append(X_split_batch.values.reshape(-1, bptt_y, 1))

        en_i = np.concatenate(en_i, axis=0)
        de_i = np.concatenate(de_i, axis=0)
        en_cat_i = np.concatenate(en_cat_i, axis=0)
        de_cat_i = np.concatenate(de_cat_i, axis=0)
        de_o = np.concatenate(de_o, axis=0)

        gc.collect()

        return en_i, en_cat_i, de_i, de_cat_i, de_o

    def create_dataset_mask(
        self,
        X,
        split_d_cols,
        all_num_cols,
        all_cat_cols,
        id_col,
        bptt_x,
        bptt_y,
        max_lag,
    ):
        en_i, de_i, en_cat_i, de_cat_i = (
            [],
            [],
            [],
            [],
        )

        X_split = X[X["d"].isin(split_d_cols)]

        for train_i in range(
            ((len(split_d_cols) - max_lag) % bptt_y) + 1,
            len(split_d_cols) - bptt_x - bptt_y + 2,
            bptt_y,
        ):
            en_i_d_cols_batch = split_d_cols[train_i : train_i + bptt_x]
            de_i_d_cols_batch = split_d_cols[
                train_i + bptt_x - 1 : train_i + bptt_x - 1 + bptt_y
            ]

            # encode_decode numeric input
            X_split_batch = X_split.loc[
                X_split["d"].isin(en_i_d_cols_batch + de_i_d_cols_batch[1:]), :
            ]
            id_cols_batch = X_split_batch.groupby([id_col])["d"].count() == len(
                en_i_d_cols_batch + de_i_d_cols_batch[1:]
            )
            X_split_batch = X_split_batch.loc[
                X_split_batch[id_col].isin(id_cols_batch[id_cols_batch].index),
                all_num_cols,
            ]

            X_split_batch = X_split_batch.values.reshape(
                -1, bptt_x + bptt_y - 1, len(all_num_cols)
            )

            en_i.append(X_split_batch[:, : len(en_i_d_cols_batch)])
            de_i.append(X_split_batch[:, len(en_i_d_cols_batch) - 1 :])

            # encode_decode categorical input
            X_split_batch = X_split.loc[
                X_split["d"].isin(en_i_d_cols_batch + de_i_d_cols_batch[1:]), :
            ]
            X_split_batch = X_split_batch.loc[
                X_split_batch[id_col].isin(id_cols_batch[id_cols_batch].index),
                all_cat_cols,
            ]
            X_split_batch = X_split_batch.values.reshape(
                -1, bptt_x + bptt_y - 1, len(all_cat_cols)
            )

            en_cat_i.append(X_split_batch[:, : len(en_i_d_cols_batch)])
            de_cat_i.append(X_split_batch[:, len(en_i_d_cols_batch) - 1 :])

        en_i = np.concatenate(en_i, axis=0)
        de_i = np.concatenate(de_i, axis=0)
        en_cat_i = np.concatenate(en_cat_i, axis=0)
        de_cat_i = np.concatenate(de_cat_i, axis=0)

        gc.collect()

        return en_i, en_cat_i, de_i, de_cat_i


@stop_watch
def setting_dataloader(
    X,
    train_d_cols,
    test_d_cols,
    all_num_cols,
    all_cat_cols,
    bptt_x,
    bptt_y,
    max_lag,
    test_size,
    batch_size,
):
    trainset = TSdatasets(
        X,
        train_d_cols[max_lag + 1 : -test_size],
        all_num_cols,
        all_cat_cols,
        bptt_x,
        bptt_y,
        max_lag,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=min(batch_size, trainset.en_i.shape[0]), shuffle=True
    )
    del trainset
    gc.collect()

    validset = TSdatasets(
        X,
        train_d_cols[-(test_size + bptt_x) :],
        all_num_cols,
        all_cat_cols,
        bptt_x,
        bptt_y,
        max_lag,
    )
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=min(batch_size, validset.en_i.shape[0]), shuffle=False
    )
    del validset
    gc.collect()

    validmaskset = TSdatasets(
        X,
        train_d_cols[-(test_size + bptt_x) :],
        all_num_cols,
        all_cat_cols,
        bptt_x,
        bptt_y,
        max_lag,
        mask=True,
    )
    validmaskloader = torch.utils.data.DataLoader(
        validmaskset,
        batch_size=min(batch_size, validmaskset.en_i.shape[0]),
        shuffle=False,
    )
    del validmaskset
    gc.collect()

    testset = TSdatasets(
        X,
        train_d_cols[-max_lag:] + test_d_cols,
        all_num_cols,
        all_cat_cols,
        bptt_x,
        bptt_y,
        max_lag,
        mask=True,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=min(batch_size, testset.en_i.shape[0]), shuffle=False
    )
    del testset
    gc.collect()

    return trainloader, validloader, validmaskloader, testloader
