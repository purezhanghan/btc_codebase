# data
import os
import pandas as pd
from datetime import datetime
import numpy as np

# models
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# args parser
import argparse

# database supports
import pymysql

# ordered dict
from collections import OrderedDict

# preprocess, default min_max
from sklearn.preprocessing import MinMaxScaler

# warnings
import warnings
warnings.simplefilter('ignore')

PRICE_NAME = ['价格']
HOLD_AMOUNT_NAME = ['100-1000', '1000-10000', '10000-100000']
# NORM_COLS = ['价格', '100-1000', '1000-10000', '10000-100000']
NORM_COLS = PRICE_NAME + HOLD_AMOUNT_NAME
SPLIT_SIZE = 0.7

MODEL_LIST = [LinearRegression, SVR, KNeighborsRegressor]

SCALER_DICT = OrderedDict()


def _sep_detect(path):
    with open(path, 'rb') as f:
        first_line = f.readline()
    if str(first_line).find(';') != -1:
        return ';'
    elif str(first_line).find(',') != -1:
        return ','
    else:
        return 'other'
        # raise ValueError(
        #    'file not separated by ";" or "," neither, please check the seperator.')


def _read_data(path):
    sep = _sep_detect(path)
    if sep != 'other':
        data = pd.read_csv(path)
    else:
        data = pd.read_csv(path, skiprows=1)
    return data


def model_iteration():
    pass


def hyper_parameter_search():
    pass


def _parse_path(base_path, keyword):
    bool_list = [keyword in s for s in os.listdir(base_path)]
    if any(bool_list):
        sub_path = os.listdir(base_path)[bool_list.index(True)]
        read_path = os.path.join(base_path, sub_path)
        return read_path
    else:
        if keyword == PRICE_NAME[0]:
            raise ValueError('price data path is invalid.')
        else:
            print('Path for auxiliary data {} is invalid'.format(keyword))
        return None


def _data_clean(key, data):
    data.columns = ['timestamp', key]
    for col in data.columns:
        try:
            data = data[data[col].apply(lambda x: x.isnumeric())]
        except AttributeError:
            print('{} has no string values in column {}'.format(key, col))

    data['timestamp'] = data['timestamp'].astype(int)
    data[key] = data[key].astype(float)
    data.loc[:, '{}_datetime'.format(key)] = data.apply(
        lambda row: pd.to_datetime(datetime.fromtimestamp(row['timestamp'])), axis=1)
    print('{} \nmax datetime: {} \nmin datetime {}\n'.format(
        key, data['{}_datetime'.format(key)].max(), data['{}_datetime'.format(key)].min()))
    data = data.sort_values(['{}_datetime'.format(key)])
    return data


def _find_nearest_pos(current_t, aux_data, aux_key):
    ts = aux_data['timestamp'] - current_t
    if (ts < 0).sum() == 0:
        return 'exceed condition'
    else:
        idx = np.argmax(ts[ts < 0])
        return aux_data.loc[idx]['{}_datetime'.format(aux_key)]


def _merge_aux_data(merged_data, aux_data, aux_key):
    merged_data.loc[:, '{}_time_nearest'.format(aux_key)] = merged_data.apply(
        lambda row: _find_nearest_pos(row['timestamp'], aux_data, aux_key), axis=1)
    merged_data = merged_data.merge(aux_data.drop('timestamp', axis=1), left_on=[
        '{}_time_nearest'.format(aux_key)], right_on=['{}_datetime'.format(aux_key)])
    return merged_data


def _normalize(df):
    global SCALER_DICT
    for col in NORM_COLS:
        scaler = MinMaxScaler()
        df['{}_norm'.format(col)] = scaler.fit_transform(
            df[col].values.reshape(-1, 1))
        SCALER_DICT[col] = scaler
    return df


def _model_tune(data, norm=True):
    if norm:
        train_col = list(data.columns[data.columns.str.contains('norm')])
    best_model = None
    best_score = -np.inf
    price_norm_name = '{}_norm'.format(PRICE_NAME[0])
    train_col.remove(price_norm_name)
    X = data[train_col]
    y = data[price_norm_name]
    split_flag = int(len(data) * SPLIT_SIZE)
    X_train, X_test, y_train, y_test = X[:split_flag], X[split_flag:], y[:split_flag], y[split_flag:]
    for model in MODEL_LIST:
        reg = model()
        reg.fit(X_train, y_train)
        # train_pred = model.predict(X_train)
        test_pred = reg.predict(X_test)
        cur_score = reg.score(X_test, y_test)
        if cur_score > best_score:
            best_score = cur_score
            # best_train_pred = train_pred
            best_test_pred = test_pred
            best_model = reg
        print(
            f"{model.__name__:22} R-squared: "
            f"{cur_score:.3f}\n"
            f"best_score: {best_score:.3f}"
        )
    return best_model, best_test_pred, y_test


def _validation(pred, target, scaler_dict):
    val_df = pd.DataFrame(data=np.hstack(
        (pred.reshape(-1, 1), target.values.reshape(-1, 1))), columns=['pred', 'target'])
    val_df['target_shift'] = val_df['target'].shift(-1)
    val_df['pred_shift'] = val_df['pred'].shift(-1)

    val_df['target_trend'] = val_df['target_shift'] - val_df['target']
    val_df['pred_trend'] = val_df['pred_shift'] - val_df['pred']
    val_df['simult'] = val_df.apply(
        lambda row: row['target_trend'] * row['pred_trend'] >= 0, axis=1)

    print(
        f"trend simultaneous acc: {val_df['simult'].sum() / len(val_df):.3f}"
    )


def train_main(args):
    # read in data
    data_dict = OrderedDict()
    for keyword in PRICE_NAME + HOLD_AMOUNT_NAME:
        read_in_path = _parse_path(args.read_path, keyword)
        if read_in_path is not None:
            cleaned_data = _data_clean(keyword, _read_data(read_in_path))
            # drop idnex necessary!
            cleaned_data = cleaned_data.reset_index(drop=True)
            data_dict[keyword] = cleaned_data

    merged_data = data_dict[PRICE_NAME[0]]

    for aux_key, aux_data in data_dict.items():
        if aux_key == PRICE_NAME[0]:
            continue
        merged_data = _merge_aux_data(
            merged_data, aux_data, aux_key)

    # normalization
    norm_data = _normalize(merged_data)

    # model
    best_model, best_test_pred, y_test = _model_tune(norm_data)

    # validation

    _validation(best_test_pred, y_test, SCALER_DICT)


# to do
def test_main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_search", default=False,
                        help='Turn on grid search on given parameters.', action='store_true')
    parser.add_argument("--read_path", default='./', help='default read path')
    parser.add_argument("--save_path", default='./', help='default save path')
    parser.add_argument("--train", help='train or eval mode', default=True)
    args = parser.parse_args()
    if args.train:
        train_main(args)
    else:
        test_main(args)
