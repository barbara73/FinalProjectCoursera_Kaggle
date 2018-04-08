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


def downcast_dtypes(df) :
    '''
        Changes column types in df.
    '''

    float_cols = [c for c in df if df[c].dtype == 'float64']
    int_cols = [c for c in df if df[c].dtype == 'int64']

    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int32)
    return df


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

    idx_item1 = df.item_price[df.item_price <= 0]
    idx_item2 = df.item_price[df.item_price > 100000]

    df.drop(df.index[idx_item2.index[0]], inplace=True)
    df.item_price.replace(to_replace=-1, value=2499.0 , inplace=True)

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


def get_target(data, col, old_name, new_name, foo) :
    '''
        this is the target to predict.
    '''
    
    gb = data.groupby(col, as_index=False).agg({'item_cnt_day':foo})
    gb = gb.rename(index=str, columns={'item_cnt_day': new_name})
    return gb


def prepare_test(test) :
    '''
        prepare minimum test set. add all the features which are added to the
        train set.
    '''
    
    test = test.assign(date_block_num = 34)
    #test.drop('ID', axis=1, inplace=True)
    test = test.astype('int32')

    test = test.assign(month = 11).astype('int32')
    test = test.assign(year = 2015).astype('int32')
    return test


def k_fold_mean(data, x, mean, col) :
    '''
        feature
    '''
    
    kf = KFold(5, shuffle=False)
    kf.get_n_splits(data)
    data = data.assign(new_name = np.nan)

    for train_ind, val_ind in kf.split(data) :
        X_tr, X_val = data.iloc[train_ind], data.iloc[val_ind]
        data.loc[val_ind, 'new_name'] = X_val[col].map(
            X_tr.groupby(col)[x].mean())

    data['new_name'].fillna(mean, inplace=True)
    return data


def loo_mean(data, x, mean, col) :
    '''
        feature
    '''
    
    loo_sum = data[col].map(data.groupby(col)[x].sum())
    loo_count = data[col].map(data.groupby(col)[x].count())
    data = data.assign(new_name = (loo_sum - data.target) / (
        loo_count - 1))

    data['new_name'].fillna(mean, inplace=True)
    return data


def smoothing_mean(data, x, mean, col) :
    '''
        feature
    '''
    
    mean_target = data.groupby(col)[x].transform('mean')
    nrows = data[col].map(data.groupby(col)[x].count())
    global_mean = mean
    alpha = 100
    data = data.assign(new_name = (mean_target*nrows+global_mean*alpha) / (
        nrows+alpha))
 
    data['new_name'].fillna(mean, inplace=True)
    return data


def expanding_mean(data, x, mean, col) :
    '''
        feature
    '''
    
    cum_sum = data.groupby(col)[x].cumsum() - data[x]
    cum_count = data.groupby(col).cumcount()
    data = data.assign(new_name = cum_sum / cum_count)

    data['new_name'].fillna(mean, inplace=True)
    return data
