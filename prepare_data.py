#from __future__ import division, print_function, unicode_literals
#import pylint
#import tox
import pandas as pd
import numpy as np
import os
from itertools import product
from sklearn.model_selection import KFold

def read_data() :
    DATA_FOLDER = 'readonly/final_project_data/'

    sales = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))
    items = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
    categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
    shops = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

    test = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))

    return sales, items, categories, shops, test


def create_df(sales, items, categories, shops) :
    df = pd.merge(sales, items, on='item_id')
    df_new = pd.merge(df, categories, on='item_category_id')
    df = pd.merge(df_new, shops, on='shop_id')

    #a = pd.to_datetime(df['date'], dayfirst=True, format='%d.%m.%Y')
    #df = df.assign(year = a.dt.year)
    #df = df.assign(month = a.dt.month)

    return df


def make_grid(sales, index_cols) :
    # For every month we create a grid from all shops/items combinations from that month
    grid = [] 
    for block_num in sales['date_block_num'].unique() :
        cur_shops = sales[sales['date_block_num']==block_num]['shop_id'].unique()
        cur_items = sales[sales['date_block_num']==block_num]['item_id'].unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

    # turn the grid into pandas dataframe
    grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

    return grid


def get_aggregated(sales, index_cols) :
    gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':'sum'})
    gb = gb.rename(index=str, columns={"item_cnt_day": "target"}).astype('int32')

    return gb


def join_to_existing(data1, data2, index_cols) :
    data = pd.merge(data1, data2, how='left', on=index_cols).fillna(0)

    return data


def prepare_test(test) :
    test = test.assign(date_block_num = 34)
    test.drop('ID', axis=1, inplace=True)
    test = test.astype('int32')

    test = test.assign(month = 11).astype('int32')
    test = test.assign(year = 2015).astype('int32')

    return test


def k_fold_mean(data) :
    kf = KFold(5, shuffle=False)
    kf.get_n_splits(data)
    data = data.assign(target_kfold = np.nan).astype('float32')

    for train_ind, val_ind in kf.split(data) :
        X_tr, X_val = data.iloc[train_ind], data.iloc[val_ind]
        data['target_kfold'].iloc[val_ind] = X_val.item_id.map(X_tr.groupby('item_id').target.mean())
 
    data['target_kfold'].fillna(0.3343, inplace=True)
    df = data[['shop_id', 'item_id', 'target_kfold']]
    
    return data, df


def loo_mean(data) :
    loo_sum = data['item_id'].map(data.groupby('item_id').target.sum())
    loo_count = data['item_id'].map(data.groupby('item_id').target.count())
    data = data.assign(target_loo = (loo_sum - data.target) / (loo_count - 1)).astype('float32')

    data['target_loo'].fillna(0.3343, inplace=True)
    df = data[['shop_id', 'item_id', 'target_loo']]
    
    return data, df


def smoothing_mean(data) :
    mean_target = data.groupby('item_id').target.transform('mean')
    nrows = data['item_id'].map(data.groupby('item_id').target.count())
    global_mean = 0.3343
    alpha = 100
    data = data.assign(target_smoothing = (mean_target*nrows+global_mean*alpha) / (nrows+alpha)).astype('float32')

    data['target_smoothing'].fillna(0.3343, inplace=True)
    df = data[['shop_id', 'item_id', 'target_smoothing']]

    return data, df


def expanding_mean(data) :
    cum_sum = data.groupby('item_id').target.cumsum() - data.target
    cum_count = data.groupby('item_id').cumcount()
    data = data.assign(target_exp_mean = cum_sum / cum_count).astype('float32')

    data['target_exp_mean'].fillna(0.3343, inplace=True)
    df = data[['shop_id', 'item_id', 'target_exp_mean']]

    return data, df
