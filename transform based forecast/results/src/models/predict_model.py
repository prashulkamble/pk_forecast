import pandas as pd
import numpy as np


def output_inverse(output, dtrans, X_ids, d_cols, diff_trans, sales_train_first):
    output = dtrans.inverse_transform(output.T).T
    output = pd.DataFrame(output, index=X_ids, columns=d_cols)
    if diff_trans:
        output.iloc[:, 0] += sales_train_first
        output = output.cumsum(1).clip(lower=0.0)

    return output


def to_submission(
    output_test,
    sales_train,
    sample_submission,
    test_d_cols,
    d2F_map,
    private_sub=False,
):
    pred_test = output_test.loc[sales_train["id"], test_d_cols]
    pred_test[test_d_cols] = pred_test[test_d_cols].clip(lower=0.0)

    pred_test.columns = [d2F_map[col] for col in pred_test.columns]
    if private_sub:
        pred_test.index = [
            v.replace("validation", "evaluation") for v in pred_test.index
        ]

    my_submission = sample_submission.copy()
    my_submission = my_submission.astype(np.float32)

    submission_id = [i for i in my_submission.index if i in pred_test.index]
    my_submission.loc[submission_id, :] = pred_test.loc[submission_id, :]

    return my_submission
