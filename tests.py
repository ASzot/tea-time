from convert_csv import convert
import os
import os.path as osp
import numpy as np
import pandas as pd
import random
from teafiles import rangen, DateTime, Duration
from data_cacher import DataCacher
import time

import matplotlib.pyplot as plt

from trading_calendars import get_calendar

from datetime import datetime, date

# Test symbol which is a plugin replacement for quantopians symbol class.
class Symbol(object):
    def __init__(self, symb):
        self.symbol = symb


    def __str__(self):
        return 'symbol[' + self.symbol + ']'

    def __lt__(self, other):
        return self.symbol < other.symbol

    def __gt__(self, other):
        return self.symbol > other.symbol


TEST_DIR = '.test-data'
CSV_DIR = osp.join(TEST_DIR, 'csv')
TEA_DIR = osp.join(TEST_DIR, 'tea')

num_files = 10
test_symbs = [Symbol(str(i)) for i in range(num_files)]
start_index = 5000

#LOG_LEVEL = DataCacher.LOG_INFO
LOG_LEVEL = 0

def create_cacher(cal):
    return DataCacher(cal, TEA_DIR, verbose=LOG_LEVEL, cache_count=5)

def create_test_files(cal, num_rows):
    if not osp.exists(CSV_DIR):
        os.makedirs(CSV_DIR)

    num_cols = 5
    # This is 2009-10-30

    for symb in test_symbs:
        all_data = []
        all_times = []
        for ts in cal.schedule.index[start_index:start_index + num_rows]:
            data = [random.random() for n in range(num_cols)]
            all_data.append(data)
            all_times.append(ts)

        col_names = ['col_%s' % col_i for col_i in range(num_cols)]

        df = pd.DataFrame(all_data, columns = col_names, index=all_times)
        df.index = pd.to_datetime(df.index)

        filename = osp.join(CSV_DIR, '%s.csv' % symb.symbol)

        df.to_csv(filename)


def test_data_cacher_single(cal, dt, window):
    cacher = create_cacher(cal)
    start = time.time()
    result = cacher.get_symbs(test_symbs[:1], dt, window)
    elapsed = time.time() - start
    print('Data cacher single', elapsed)

    return elapsed


def test_pd_single(dt, window):
    use_dt = dt.date().strftime('%Y-%m-%d')

    start = time.time()
    df = pd.read_csv(CSV_DIR + '/' + test_symbs[0].symbol + '.csv', index_col = 0, parse_dates=True)
    i = df.index.get_loc(use_dt)
    result = df.iloc[i - window:i]
    elapsed = time.time() - start

    print('PD single', elapsed)

    return elapsed


def test_data_cacher_multiple(cal, dt, window):
    cacher = create_cacher(cal)
    start = time.time()
    results = cacher.get_symbs(test_symbs, dt, window)
    elapsed = time.time() - start

    print('Data cacher multiple', elapsed)

    return elapsed


def test_pd_multiple(dt, window):
    use_dt = dt.date().strftime('%Y-%m-%d')

    start = time.time()
    results = {}
    for symb in test_symbs:
        df = pd.read_csv(CSV_DIR + '/' + test_symbs[0].symbol + '.csv', index_col = 0, parse_dates=True)
        i = df.index.get_loc(use_dt)
        result = df.iloc[i - window:i]
        results[symb.symbol] = result

    results = pd.Panel.from_dict(results)
    elapsed = time.time() - start
    print('PD multiple', elapsed)

    return elapsed


def test_data_cacher_random_access(cal, dt, other_dt, window):
    cacher = create_cacher(cal)
    results = cacher.get_symbs(test_symbs, dt, window)

    start = time.time()
    results = cacher.get_symbs(test_symbs, other_dt, window)

    elapsed = time.time() - start
    print('Data cacher random access', elapsed)

    return elapsed


def test_pd_random_access(dt, other_dt, window):
    use_dt = dt.date().strftime('%Y-%m-%d')
    use_other_dt = other_dt.date().strftime('%Y-%m-%d')

    results = {}
    dfs = {}
    for symb in test_symbs:
        df = pd.read_csv(CSV_DIR + '/' + test_symbs[0].symbol + '.csv', index_col = 0, parse_dates=True)
        dfs[symb.symbol] = df
        i = df.index.get_loc(use_dt)
        result = df.iloc[i - window:i]
        results[symb.symbol] = result

    results = pd.Panel.from_dict(results)

    start = time.time()
    other_results = {}
    for symb in test_symbs:
        i = dfs[symb.symbol].index.get_loc(use_other_dt)

        result = results[symb.symbol].iloc[i - window:i]
        other_results[symb.symbol] = result

    other_results = pd.Panel.from_dict(other_results)

    elapsed = time.time() - start
    print('PD random access', elapsed)

    return elapsed

cal = get_calendar('NYSE')
#create_test_files(cal, 4000)
#convert(CSV_DIR, TEA_DIR)
dt = cal.schedule.index[start_index + 50]

window = 30


trails = 3
elapsed_pd_single = []
elapsed_dc_single = []

elapsed_pd_mult = []
elapsed_dc_mult = []

elapsed_pd_ra_mem = []
elapsed_dc_ra_mem = []

elapsed_pd_ra_seq = []
elapsed_dc_ra_seq = []

elapsed_pd_ra = []
elapsed_dc_ra = []


for i in range(trails):
    elapsed_pd_single.append(test_pd_single(dt, window))
    elapsed_dc_single.append(test_data_cacher_single(cal, dt, window))

    elapsed_pd_mult.append(test_pd_multiple(dt, window))
    elapsed_dc_mult.append(test_data_cacher_multiple(cal, dt, window))


    # Random access in memory
    other_dt = cal.schedule.index[start_index + 54]
    elapsed_pd_ra_mem.append(test_pd_random_access(dt, other_dt, window))
    elapsed_dc_ra_mem.append(test_data_cacher_random_access(cal, dt, other_dt, window))


    # Random access sequential
    other_dt = cal.schedule.index[start_index + 55]
    elapsed_pd_ra_seq.append(test_pd_random_access(dt, other_dt, window))
    elapsed_dc_ra_seq.append(test_data_cacher_random_access(cal, dt, other_dt, window))

    # Random access out of memory non-sequential
    other_dt = cal.schedule.index[start_index + 80]
    elapsed_pd_ra.append(test_pd_random_access(dt, other_dt, window))
    elapsed_dc_ra.append(test_data_cacher_random_access(cal, dt, other_dt, window))


index = np.arange(5)
pd_means = [np.mean(elapsed_pd_single), np.mean(elapsed_pd_mult),
        np.mean(elapsed_pd_ra_mem), np.mean(elapsed_pd_ra_seq),
        np.mean(elapsed_pd_ra)]

dc_means = [np.mean(elapsed_dc_single), np.mean(elapsed_dc_mult),
        np.mean(elapsed_dc_ra_mem), np.mean(elapsed_dc_ra_seq),
        np.mean(elapsed_dc_ra)]


bar_width = 0.3

plt.bar(index, pd_means, bar_width, color='b', label='Pandas')
plt.bar(index + bar_width, dc_means, bar_width, color='r', label='Tea Time')

plt.xlabel('test')
plt.ylabel('Times (seconds)')
plt.legend()
plt.xticks(index, ('Single', 'Multi', 'In Memory', 'Sequential', 'Out memory'))
plt.savefig('bench.png')
