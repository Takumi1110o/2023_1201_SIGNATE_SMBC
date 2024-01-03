import pandas as pd
import numpy as np
import category_encoders as ce

def LeaveOneOut(length, colum_list, df):
    target_col= 'health'
    train_df = df[:length]
    test_df  = df[length:]

    for co in colum_list:
        cate_col = f'target_{co}'
        loo = ce.LeaveOneOutEncoder(cols=co, random_state=42)
        train_df[cate_col] = loo.fit_transform(train_df[co], train_df[target_col])
        test_df[cate_col] = loo.transform(test_df[co])

    data = pd.concat([train_df, test_df])
    return data

def CATBOOST(length, colum_list, df, yml):
    target_col = yml['target_col']
    train_df = df[:length]
    test_df  = df[length:]

    for co in colum_list:
        cate_col = f'target_{co}'
        cbe = ce.CatBoostEncoder(cols=co, random_state=yml['params']['seed'])
        train_df[cate_col] = cbe.fit_transform(train_df[co], train_df[target_col])
        test_df[cate_col] = cbe.transform(test_df[co])

    data = pd.concat([train_df, test_df])
    return data

def JAMESSTEIN(length, colum_list, df, yml):
    target_col = yml['target_col']
    train_df = df[:length]
    test_df  = df[length:]

    for co in colum_list:
        cate_col = f'target_{co}'
        cbe = ce.JamesSteinEncoder(cols=co, random_state=yml['params']['seed'])
        train_df[cate_col] = cbe.fit_transform(train_df[co], train_df[target_col])
        test_df[cate_col] = cbe.transform(test_df[co])

    data = pd.concat([train_df, test_df])
    return data