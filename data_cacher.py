from teafiles import TeaFile, DateTime
import os
import os.path as osp
import numpy as np
import pandas as pd

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DAILY_TIME_FORMAT = '%Y-%m-%d'

def mk_str(s, daily=False):
    if daily:
        return s.strftime(DAILY_TIME_FORMAT)
    else:
        return s.strftime(TIME_FORMAT)


class DataCacher(object):
    LOG_DEBUG = 2
    LOG_INFO = 1

    def __init__(self, cal, cache_loc, verbose=0, cache_count = 400):
        self.cal = cal
        # The dates are synced up between all of these dates
        self.__cache = {}
        self.__pos_lookup = {}
        self.__dates = []
        self.__tea_files = {}
        self.__cache_loc = cache_loc

        self.__cache_count = cache_count
        # Number of time steps that we should keep something that hasn't been
        # used in a while
        self.__keep_length = 20
        self.__cols = None

        self.__unused = {}
        self.__verbose = verbose

    def __log(self, msg, mode):
        if self.__verbose >= mode:
            print('[cacher] ' + msg)

    def __remove_key(self, symb):
        self.__tea_files[symb].close()
        del self.__tea_files[symb]
        del self.__cache[symb]


    def get_cols(self):
        if self.__cols is None:
            first_key = next(iter(self.__tea_files))
            self.__cols= self.__tea_files[first_key].description.itemdescription.fields
        return self.__cols

    def __add_data_search(self, symb, tf, start_dt, check_dates, lookahead):
        found_idx = tf.seek_time_index(DateTime.from_pandas_ts(start_dt))
        if found_idx is None:
            self.__remove_key(symb)
            return False

        for i in range(lookahead):
            row = tf.read()
            if row is None or not row.time.is_equal_to_ts(check_dates[i]):
                row = [np.nan] * (len(self.get_cols()))
            self.__cache[symb].append(list(row[1:]))

        return True

    def __add_data(self, symb, tf, check_dates, lookahead, row):
        for i in range(lookahead):
            if row is None or not row.time.is_equal_to_ts(check_dates[i]):
                row = [np.nan] * (len(self.get_cols()))

            self.__cache[symb].append(list(row[1:]))
            row = tf.read()

    def __refresh(self, symbs, dt, lookbehind):
        idx = self.cal.schedule.index.get_loc(dt)

        self.__log('Refreshing cache', DataCacher.LOG_INFO)

        self.__log('Working date %s at calendar index %i' % (str(dt), idx),
                DataCacher.LOG_DEBUG)

        valid_symbs = []

        for symb in symbs:
            if symb in self.__cache:
                valid_symbs.append(symb)

        self.__log('Valid symbols %s' % str(valid_symbs), DataCacher.LOG_DEBUG)

        for symb in valid_symbs:
            tf = self.__tea_files[symb]

            row = tf.read()

            # Get only the day part of the time
            cur_day = str(row.time).split(' ')[0]

            # Sequential reads could be optimized
            #if cur_day != dt:
            self.__cache[symb] = []

            lookahead = lookbehind + self.__cache_count
            start_idx = idx - lookbehind
            check_dates = list(self.cal.schedule.index[start_idx:start_idx +
                    lookahead])
            start_dt = check_dates[0]
            lookahead = len(check_dates)

            self.__add_data_search(symb, tf, start_dt, check_dates, lookahead)
            #else:
            #    self.__log('Fetching sequentailly', DataCacher.LOG_INFO)
            #    # Minus one because we need to append the row we just read.
            #    self.__cache[symb] = self.__cache[symb][-(lookbehind - 1):]

            #    lookahead = self.__cache_count
            #    check_dates = list(self.cal.schedule.index[idx-lookbehind:idx +
            #            lookahead])

            #    self.__add_data(symb, tf, check_dates, lookahead, row)

            if symb in self.__cache and np.isnan(self.__cache[symb][:lookbehind]).any():
                self.__remove_key(symb)

        exists = len(self.__cache) > 0
        if exists:
            self.__dates = self.__dates[:-len(check_dates)]
            self.__dates.extend(check_dates)
        else:
            lookahead = lookbehind + self.__cache_count
            start_idx = idx - lookbehind
            self.__dates = list(self.cal.schedule.index[start_idx:start_idx +
                    lookahead])

        self.__build_new_symbs(symbs, lookbehind)

        self.__pos_lookup = {}
        for i, time in enumerate(self.__dates):
            self.__pos_lookup[mk_str(time, daily=True)] = i

    def __build_new_symbs(self, symbs, lookbehind):
        new_symbs = []

        unmarked_tea_files = {symb: True for symb in self.__tea_files}

        for symb in symbs:
            if symb in unmarked_tea_files:
                del unmarked_tea_files[symb]

            if symb not in self.__cache:
                new_symbs.append(symb)

        self.__log('New symbols %s' % str(new_symbs), DataCacher.LOG_DEBUG)

        for symb in unmarked_tea_files:
            if symb not in self.__unused:
                self.__unused[symb] = 0
            else:
                self.__unused[symb] += 1

        remove_keys = []
        for symb in self.__unused:
            if self.__unused[symb] > self.__keep_length:
                self.__remove_key(symb)
                remove_keys.append(symb)
            else:
                valid_symbs.append(symb)

        for remove_key in remove_keys:
            del self.__unused[remove_key]

        for symb in new_symbs:
            symb_path = osp.join(self.__cache_loc, symb.symbol + '.tea')
            self.__tea_files[symb] = TeaFile.openread(symb_path)
            self.__cache[symb] = []

            tf = self.__tea_files[symb]

            start_dt = self.__dates[0]
            lookahead = len(self.__dates)

            self.__add_data_search(symb, tf, start_dt, self.__dates, lookahead)
            if symb in self.__cache and np.isnan(self.__cache[symb][:lookbehind]).any():
                self.__remove_key(symb)

    def get_symbs(self, symbs, dt, lookback):
        assert lookback > 0, 'Lookback must be greater than 0. If you want only one bar of data set it to 1.'

        localized_dt = dt.tz_convert('UTC').tz_localize(None)
        self.__log('Fetching date for %s' % (localized_dt), DataCacher.LOG_INFO)

        # Just convert to daily. This may need to be fixed to make this more
        # extendible to more time ranges.
        use_dt = localized_dt

        use_dt = mk_str(use_dt, daily=True)
        # Check if we will need a refresh.
        if use_dt not in self.__pos_lookup or self.__pos_lookup[use_dt] - lookback < 0:
            self.__refresh(symbs, use_dt, lookback)
        else:
            assert len(self.__cache) > 0
            self.__build_new_symbs(symbs, lookback)

        all_symb_dfs = {}
        start_loc = self.__pos_lookup[use_dt]
        for symb in symbs:
            self.__log('Date exists', DataCacher.LOG_INFO)
            if symb not in self.__cache:
                # Symbol has invalid data.
                continue
            symb_cache = self.__cache[symb]

            df_data = np.array(symb_cache[start_loc - lookback:start_loc])
            use_cols = [c.name for c in self.get_cols()]
            use_dates = self.__dates[start_loc - lookback:start_loc]
            all_symb_dfs[symb] = pd.DataFrame(df_data, index=use_dates,
                    columns=use_cols[1:])

        result = pd.Panel.from_dict(all_symb_dfs)
        # Remove any time zone buisness
        ####################################################
        # Redundant check to make sure the data is valid.

        # Look ahead bias.
        for check_date in result.major_axis:
            check_date_str = mk_str(check_date, daily=True)
            if check_date_str > use_dt:
                raise ValueError('Look ahead bias')
            elif check_date_str == use_dt:
                raise ValueError('Data for the query data exists. Look ahead bias could exist')

        idx = self.cal.schedule.index.get_loc(use_dt)

        check_dates = self.cal.schedule.index[idx - lookback:idx]

        for check_date, result_date in zip(check_dates, list(result.major_axis)):
            if check_date != result_date:
                print('Check dates', list(check_dates))
                print('Resulting dates', list(result.major_axis))
                raise ValueError('Dates do not match')
        ####################################################

        if len(result.items) != 0:
            result.major_axis = result.major_axis.tz_localize(None)

        return result

