import pandas as pd
import os
from datetime import datetime

base_dir = '/Users/Han/Downloads/data/'
pos1 = pd.read_csv(os.path.join(base_dir, '持有[100-1000]范围内比特币总量.csv'), names=['timestamp', 'amount'])
pos1.drop(0, inplace=True)
pos1['timestamp'] = pos1['timestamp'].astype(int)

pos1.loc[:, 'datetime'] = pos1.apply(lambda row: pd.to_datetime(datetime.fromtimestamp(row['timestamp'])), axis=1)
print(pos1.datetime.max())
print(pos1.datetime.min())
print(pos1.datetime)

price = pd.read_csv(os.path.join(base_dir, '价格.csv'), names=['timestamp', 'price'])
price.drop(0, inplace=True)
price['timestamp'] = price['timestamp'].astype(int)

price.loc[:, 'datetime'] = price.apply(lambda row: pd.to_datetime(datetime.fromtimestamp(row['timestamp'])), axis=1)
print(price.head())
print(price.datetime.max())
print(price.datetime.min())



