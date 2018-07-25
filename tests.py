from convert_csv import convert
import os
import os.path as osp
import numpy as np
import pandas as pd
import random
from teafiles import rangen, DateTime, Duration
from data_cacher import DataCacher

TEST_DIR = '.test-data'
CSV_DIR = osp.join(TEST_DIR, 'csv')
TEA_DIR = osp.join(TEST_DIR, 'tea')

def create_test_files():
    if not osp.exists(CSV_DIR):
        os.makedirs(CSV_DIR)

    num_files = 1
    num_cols = 5
    num_rows = 200

    for i in range(num_files):
        all_data = []
        all_times = []
        for t in rangen(DateTime(2000, 1, 1), Duration(minutes=1), num_rows):
            data = [random.random() for n in range(num_cols)]
            all_data.append(data)
            ts = pd.to_datetime(t.ticks/ 1e3, unit='s')
            all_times.append(ts)

        col_names = ['col_%s' % col_i for col_i in range(num_cols)]

        df = pd.DataFrame(all_data, columns = col_names, index=all_times)
        df.index = pd.to_datetime(df.index)

        filename = osp.join(CSV_DIR, '%i.csv' % i)

        df.to_csv(filename)

create_test_files()
convert(CSV_DIR, TEA_DIR)

