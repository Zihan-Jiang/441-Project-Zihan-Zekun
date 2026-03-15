FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "hl_range",
    "oc_return",
    "ma_5_ratio",
    "ma_10_ratio",
    "vol_5",
    "vol_10",
    "vol_chg_1",
    "vol_ratio_5",
]

TARGET_COLUMN = "Target"

DEFAULT_TRAIN_END = "2020-12-31"
DEFAULT_VAL_END = "2022-12-31"

MODEL_NAMES = [
    "logistic_regression",
    "svm_linear",
    "svm_rbf",
    "random_forest",
    "gradient_boosting",
]