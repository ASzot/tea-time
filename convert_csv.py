import sys
import os
import os.path as osp
import pandas as pd
from teafiles import TeaFile, DateTime
import numpy as np

from tqdm import tqdm


def convert(convert_folder_name, result_folder_name, column_rename=None):
    if not osp.exists(result_folder_name):
        os.makedirs(result_folder_name)

    for symb_file in tqdm(os.listdir(convert_folder_name)):
        symb_name = symb_file.split('.')[0]
        result_filename = osp.join(result_folder_name, symb_name + '.tea')

        df = pd.read_csv(osp.join(convert_folder_name, symb_file), index_col =
                0, parse_dates=True)

        if column_rename is not None:
            df = df.rename(columns = column_rename)

        df_cols = df.columns
        cols = 'time ' + ' '.join(df_cols)

        mapped_type = ['q' if t == np.int64 else 'd' for t in df.dtypes]
        types = 'q' + ''.join(mapped_type)

        USE_DECIMALS = 3

        with TeaFile.create(result_filename, cols, types, symb_name, {"decimals": USE_DECIMALS}) as tf:
            for index, row in df.iterrows():
                pd_dt = row.name
                dt = DateTime(pd_dt.year, pd_dt.month, pd_dt.day, pd_dt.hour, pd_dt.minute, pd_dt.second)

                select_data = [round(row[col], USE_DECIMALS) if types[i + 1] == 'd' else int(row[col])
                        for i, col in enumerate(df_cols)]

                tf.write(dt, *select_data)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Invalid number of command line arguments')

    convert_folder_name = sys.argv[1]
    result_folder_name = sys.argv[2]

    convert(convert_folder_name, result_folder_name)
    #convert(convert_folder_name, result_folder_name, {
    #    'time': 'open',
    #    'open': 'high',
    #    'high': 'low',
    #    'low': 'close',
    #    'close': 'volume'
    #})
