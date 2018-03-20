import pandas as pd
import numpy as np
import os
from itertools import product
from sklearn.model_selection import KFold

def read_data() :
    '''
        read the data from csv.
    '''
    
    DATA_FOLDER = 'readonly/final_project_data/'

    sales = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv'))
    items = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
    categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
    shops = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

    test = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))
    return sales, items, categories, shops, test


def create_df(sales, items, categories, shops) :
    '''
        create data frame by intersecting the 4 df's, which is then needed
        for feature extraction.
    '''
  
    df = pd.merge(sales, items, on='item_id')
    df_new = pd.merge(df, categories, on='item_category_id')
    df = pd.merge(df_new, shops, on='shop_id')

    a = pd.to_datetime(df['date'], dayfirst=True, format='%d.%m.%Y')
    df = df.assign(year = a.dt.year)
    df = df.assign(month = a.dt.month)
    return df


def make_grid(sales, index_cols) :
    '''
        make a grid to get all the possibilities of the items and shops.
    '''
    
    grid = [] 
    for block_num in sales['date_block_num'].unique() :
        cur_shops = sales[sales['date_block_num']==block_num]['shop_id'].unique()
        cur_items = sales[sales['date_block_num']==block_num]['item_id'].unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

    grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
    return grid


def get_target(data, index_cols) :
    '''
        this is the target to predict.
    '''
    
    gb = data.groupby(index_cols,as_index=False).agg({'item_cnt_day':'sum'})
    gb = gb.rename(index=str, columns={"item_cnt_day": "target"})
    return gb


def count_categories(data, index_cols) :
    '''
        feature
    '''
    
    gb = data.groupby(index_cols,as_index=False).agg({'item_category_id':'count'})
    gb = gb.rename(index=str, columns={"item_category_id": "count_category"})
    return gb


def join_to_existing(data1, data2, index_cols, mean) :
    '''
        join new features to existing df.
    '''
    
    data = pd.merge(data1, data2, how='left', on=index_cols).fillna(mean)
    return data


def prepare_test(test) :
    '''
        prepare minimum test set. add all the features which are added to the
        train set.
    '''
    
    test = test.assign(date_block_num = 34)
    test.drop('ID', axis=1, inplace=True)
    test = test.astype('int32')

    test = test.assign(month = 11).astype('int32')
    test = test.assign(year = 2015).astype('int32')
    return test


def k_fold_mean(data, x, mean) :
    '''
        feature
    '''
    
    kf = KFold(5, shuffle=False)
    kf.get_n_splits(data)
    data = data.assign(new_name = np.nan).astype('float32')

    for train_ind, val_ind in kf.split(data) :
        X_tr, X_val = data.iloc[train_ind], data.iloc[val_ind]
        data['new_name'].iloc[val_ind] = X_val.item_id.map(
            X_tr.groupby('item_id')[x].mean())

    data['new_name'].fillna(mean, inplace=True)
    df = data[['shop_id', 'item_id', 'new_name']]
    return data, df


def loo_mean(data, x, mean) :
    '''
        feature
    '''
    
    loo_sum = data['item_id'].map(data.groupby('item_id')[x].sum())
    loo_count = data['item_id'].map(data.groupby('item_id')[x].count())
    data = data.assign(new_name = (loo_sum - data.target) / (
        loo_count - 1)).astype('float32')

    data['new_name'].fillna(mean, inplace=True)
    df = data[['shop_id', 'item_id', 'new_name']]
    return data, df


def smoothing_mean(data, x, mean) :
    '''
        feature
    '''
    
    mean_target = data.groupby('item_id')[x].transform('mean')
    nrows = data['item_id'].map(data.groupby('item_id')[x].count())
    global_mean = mean
    alpha = 100
    data = data.assign(new_name = (mean_target*nrows+global_mean*alpha) / (
        nrows+alpha)).astype('float32')
 
    data['new_name'].fillna(mean, inplace=True)
    df = data[['shop_id', 'item_id', 'new_name']]
    return data, df


def expanding_mean(data, x, mean) :
    '''
        feature
    '''
    
    cum_sum = data.groupby('item_id')[x].cumsum() - data[x]
    cum_count = data.groupby('item_id').cumcount()
    data = data.assign(new_name = cum_sum / cum_count).astype('float32')

    data['new_name'].fillna(mean, inplace=True)
    df = data[['shop_id', 'item_id', 'new_name']]
    return data, df
