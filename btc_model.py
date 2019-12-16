# data
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sys

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

# pickle
import pickle

# logging
import logging.config
import logging

# yml
import yaml


warnings.simplefilter('ignore')

PRICE_NAME = ['价格']
HOLD_AMOUNT_NAME = ['100-1000', '1000-10000', '10000-100000']
# NORM_COLS = ['价格', '100-1000', '1000-10000', '10000-100000']
SPLIT_SIZE = 0.7

MODEL_LIST = [LinearRegression, SVR, KNeighborsRegressor]


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


def _parse_path(base_path, keyword, train):
    bool_list = [keyword in s for s in os.listdir(base_path)]
    if any(bool_list):
        sub_path = os.listdir(base_path)[bool_list.index(True)]
        read_path = os.path.join(base_path, sub_path)
        return read_path
    else:
        if keyword == PRICE_NAME[0] and not train:
            raise ValueError('price data path is invalid, exit')
        else:
            logger.warn(
                'Path for auxiliary data {} is invalid\n'.format(keyword))
        return None


def _data_clean(key, data):
    data.columns = ['timestamp', key]
    for col in data.columns:
        try:
            data = data[data[col].apply(lambda x: x.isnumeric())]
        except AttributeError:
            logger.debug(
                '{} has no string values in column {}\n'.format(key, col))

    data['timestamp'] = data['timestamp'].astype(int)
    data[key] = data[key].astype(float)
    data.loc[:, '{}_datetime'.format(key)] = data.apply(
        lambda row: pd.to_datetime(datetime.fromtimestamp(row['timestamp'])), axis=1)
    logger.info('{} max datetime: {}'.format(
        key, data['{}_datetime'.format(key)].max()))
    logger.info('{} min datetime: {}\n'.format(
        key, data['{}_datetime'.format(key)].min()))
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
    merged_data = merged_data.loc[merged_data['{}_time_nearest'.format(
        aux_key)] != 'exceed condition']
    merged_data['{}_time_nearest'.format(aux_key)] = pd.to_datetime(
        merged_data['{}_time_nearest'.format(aux_key)])
    aux_data['{}_datetime'.format(aux_key)] = pd.to_datetime(
        aux_data['{}_datetime'.format(aux_key)])
    merged_data = merged_data.merge(aux_data.drop('timestamp', axis=1), left_on=[
        '{}_time_nearest'.format(aux_key)], right_on=['{}_datetime'.format(aux_key)])
    return merged_data


def _normalize(df, scaler_dict, is_training):
    if is_training:
        for col in PRICE_NAME + HOLD_AMOUNT_NAME:
            scaler = MinMaxScaler()
            df['{}_norm'.format(col)] = scaler.fit_transform(
                df[col].values.reshape(-1, 1))
            scaler_dict[col] = scaler
    else:
        for col in HOLD_AMOUNT_NAME:
            try:
                df['{}-test_norm'.format(col)] = scaler_dict[col].transform(
                    df['{}-test'.format(col)].values.reshape(-1, 1))
            except ValueError:
                logger.error(
                    f"There is no scaler prepared for {col} during test process, exit system\n")
                sys.exit(1)
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
        logger.info(
            f"{model.__name__:22} R-squared: "
            f"{cur_score:.3f}\t"
            f"best_score: {best_score:.3f}"
        )
    return best_model, best_test_pred, y_test


def _validation(pred, target):
    val_df = pd.DataFrame(data=np.hstack(
        (pred.reshape(-1, 1), target.values.reshape(-1, 1))), columns=['pred', 'target'])
    val_df['target_shift'] = val_df['target'].shift(-1)
    val_df['pred_shift'] = val_df['pred'].shift(-1)

    val_df['target_trend'] = val_df['target_shift'] - val_df['target']
    val_df['pred_trend'] = val_df['pred_shift'] - val_df['pred']
    val_df['simult'] = val_df.apply(
        lambda row: row['target_trend'] * row['pred_trend'] >= 0, axis=1)

    logger.info(
        f"trend simultaneous acc: {val_df['simult'].sum() / len(val_df):.3f}"
    )


def train_main(args):
    # scaler_dict
    scaler_dict = OrderedDict()
    # read in data
    data_dict = OrderedDict()
    for keyword in PRICE_NAME + HOLD_AMOUNT_NAME:
        read_in_path = _parse_path(args.read_path, keyword, args.train)
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
    norm_data = _normalize(merged_data, scaler_dict, args.train)

    # model
    best_model, best_test_pred, y_test = _model_tune(norm_data)

    # validation

    _validation(best_test_pred, y_test)

    # if save
    save_path = args.save_path + '/' + \
        str(datetime.today().replace(microsecond=0)) + '_saved_model_v1.pkl'
    character_map = {
        ord(':'): '_',
        ord('-'): '_',
        ord(' '): '_'
    }
    save_path = save_path.translate(character_map)
    save_dict = {'scaler': scaler_dict, 'model': best_model}
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)
        logger.info(f"save model in path {type(best_model).__name__}")


# to do
def test_main(args):
    model_path = max([i for i in os.listdir(args.save_path) if 'pkl' in i])
    try:
        with open(os.path.join(args.save_path, model_path), 'rb') as f:
            read_dict = pickle.load(f)
    except pickle.UnpicklingError:
        logger.error(
            'model cannot be loaded with path {}, check if there is any non-pickle exited in data folder. Remove them if necessary'.format(model_path))
        sys.exit(1)

    model = read_dict['model']
    scaler_dict = read_dict['scaler']

    data_dict = OrderedDict()
    for keyword in HOLD_AMOUNT_NAME:
        keyword = keyword + '-test'
        read_in_path = _parse_path(args.read_path, keyword, args.train)
        if read_in_path is not None:
            cleaned_data = _data_clean(keyword, _read_data(read_in_path))
            # drop idnex necessary!
            cleaned_data = cleaned_data.reset_index(drop=True)
            data_dict[keyword] = cleaned_data

    # align to 100-1000 as main timeline
    merged_data = data_dict[HOLD_AMOUNT_NAME[0] + '-test']
    for aux_key, aux_data in data_dict.items():
        if aux_key == HOLD_AMOUNT_NAME[0] + '-test':
            continue
        merged_data = _merge_aux_data(
            merged_data, aux_data, aux_key)

    # normalization
    norm_data = _normalize(merged_data, scaler_dict, args.train)

    # prediction
    norm_list = [i for i in norm_data.columns if 'norm' in i]
    pred = model.predict(norm_data[norm_list])
    pred_inverse = scaler_dict[PRICE_NAME[0]
                               ].inverse_transform(pred.reshape(-1, 1))
    pred_df = norm_data[['{}-test_datetime'.format(HOLD_AMOUNT_NAME[0])]]
    pred_df['pred_price'] = pred_inverse.flatten()
    # hold amount time + 3h
    pred_df.loc[:, 'price_datetime'] = pred_df.apply(
        lambda row: row['{}-test_datetime'.format(HOLD_AMOUNT_NAME[0])] + timedelta(hours=3), axis=1)
    pred_df.columns = ['t_datetime', 't1_price', 't1_datetime']
    pred_df = pred_df.sort_values('t_datetime')
    pred_df['pred_time'] = datetime.today().replace(microsecond=0)
    if args.t_next:
        logger.info(
            f'current time: {pred_df.iloc[-1]["t_datetime"]}\nnext time: {pred_df.iloc[-1]["t1_datetime"]}\nprice prediction: {pred_df.iloc[-1]["t1_price"]}')
        pred_df.iloc[-1].to_csv(args.read_path + '/pred.csv', mode='a')
    else:
        pred_df.to_csv(args.read_path + '/pred.csv', mode='a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_search", default=False,
                        help='Turn on grid search on given parameters.', action='store_true')
    parser.add_argument("--read_path", default='./',
                        help='default read path')
    parser.add_argument("--save_path", default='./',
                        help='default save path')
    parser.add_argument(
        "--train", help='train or eval mode', action='store_true')
    parser.add_argument(
        "--t_next", help='predict next time price only', action='store_true')
    args = parser.parse_args()

    # load logger config
    with open('./logging.yml', 'rt') as f:
        config = yaml.safe_load(f.read())
        f.close()
    logging.config.dictConfig(config)
    logger = logging.getLogger(__name__)
    logger.info("Contest is starting")
    if args.train:
        train_main(args)
    else:
        test_main(args)
