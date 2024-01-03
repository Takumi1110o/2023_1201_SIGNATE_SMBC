import pandas as pd
import numpy as np
import re
import os
import yaml
import datetime
import unicodedata
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from target_encording import LeaveOneOut, CATBOOST, JAMESSTEIN
from fill import Fill
from shape import ChangeTheShape


def process(yml):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    file_path = os.path.join(parent_dir, 'data')
    train = pd.read_csv(file_path + '/train.csv', index_col=0)
    test = pd.read_csv(file_path + '/test.csv', index_col=0)
    
    # testデータにないデータを消す
    train = train[train['spc_common']!='Himalayan cedar'] #1つ
    train = train[train['spc_common']!='Chinese chestnut'] # 3つ
    train = train[train['nta']!='MN20'] # 1つ
    train = train[train['nta']!='MN21'] # 5つ
    train = train[train['nta']!='BK27'] # 2つ
    train_co_list = train['boro_ct'].unique()
    test_co_list = test['boro_ct'].unique()
    for co in train_co_list:
        if co not in test_co_list:
            train = train[train['boro_ct']!=co]
    train.reset_index(drop=True, inplace=True)

    # データ結合
    length = len(train)
    data = pd.concat([train, test])

    # 欠損値補完
    data['steward'].fillna(0, inplace=True)
    data['problems'].fillna('0', inplace=True)
    # fill = Fill()
    shape = ChangeTheShape(data)
    encoder = LabelEncoder()

    # created_atを最小値を基準に日数データに変換
    data['created_at'] = data['created_at'].apply(lambda x: shape.change_created_at(x))
    # curb_locを1,0に変換
    data.replace({'curb_loc': {'OnCurb': 1, 'OffsetFromCurb': 0}}, inplace=True)
    # stewardは順序付けできそうなので変換
    data.replace({'steward': {'1or2': 1, '3or4': 2, '4orMore': 3}}, inplace=True)
    # 順序付け難しいのでとりあえずラベルエンコーディング
    data['guards'] = encoder.fit_transform(data['guards'])
    # 2値なので0,1に変換
    data.replace({'sidewalk': {'Damage': 1, 'NoDamage': 0}}, inplace=True)
    # 順序付け難しいのでとりあえずラベルエンコーディング
    data['user_type'] = encoder.fit_transform(data['user_type'])
    # # problemsをワンほっとエンコード
    # problems_list = ['Stones', 'BranchOther', 'BranchLights',
    #             'RootOther', 'TrunkOther', 'TrunkLights',
    #             'MetalGrates', 'WiresRope', 'Sneakers']
    # for problem in problems_list:
    #     name = 'problem_' + problem
    #     data[name] = data['problems'].apply(lambda x: 1 if problem in x else 0)
    # problemsは問題の数でエンコード
    data['problems'] = data['problems'].apply(lambda x: shape.change_problems(x))
    # 数が多すぎるやつはラベルエンコード
    # data['spc_common'] = encoder.fit_transform(data['spc_common'])
    # data['nta'] = encoder.fit_transform(data['nta'])

    # # その他カテゴリ変数をターゲットエンコーディング
    object_colums = ['spc_common', 'nta']
    if yml['target_encoding_type'] == 'jame':
        data = JAMESSTEIN(length, object_colums, data, yml)
    elif yml['target_encoding_type'] == 'cat':
        data = CATBOOST(length, object_colums, data, yml)

    # spc_latinはspc_commonと同じ、nta_nameとzip_cityはntaと同じ、boronameはborocodeと同じだから消す
    data = data.select_dtypes(include=[int, float])
    # data = data.select_dtypes(include=[int, float, 'category']).drop("id", axis=1)
    train = data[:length]
    test  = data[length:].drop(yml['target_col'], axis=1)

    return train, test


def under_sampling(train, zero_num, one_num, two_num):
    strategy = {0:zero_num, 1:one_num, 2:two_num}
    rus = RandomUnderSampler(random_state=0, sampling_strategy = strategy)
    data_resampled, y_resampled = rus.fit_resample(train, train['health'])
    return data_resampled, y_resampled


def submit(pred, score:str, logs, yml) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    file_path = os.path.join(parent_dir, 'data')
    submission = pd.read_csv(file_path + '/sample_submission.csv', header=None)
    submission.drop(1, axis=1, inplace=True)
    submission["pred"] = pred
    
    now = datetime.datetime.now()
    save_path = f'{yml["name"]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}'
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    submission.to_csv(save_path + f'/{yml["name"]}_{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv', index=False, header=False)
    
    with open(save_path + '/params.yaml', 'w') as file:
        yaml.dump(yml, file)
    
    f = open(save_path + f'/{score}_' + 'logging.txt', 'w')
    f.write(str(logs))
    f.close()
    
    return save_path

def fair_process(train, test, yml):
    length = len(train)
    data = pd.concat([train, test])

    # 0.329を超えてるやつ
    data['created_at*st_assem'] = data['created_at'] * data['st_assem']
    data['st_assem*cncldist'] = data['st_assem'] * data['cncldist']
    data['problems/cb_num'] = data['problems'] / data['cb_num']
    data['cb_num/cncldist'] = data['cb_num'] / data['cncldist']
    data['st_assem/st_senate'] = data['st_assem'] / data['st_senate']
    
    # border_score_us_fairが0.325、border_score_us_poorが0.385を両方超えたやつ
    data['tree_dbh*boro_ct'] = data['tree_dbh'] * data['boro_ct']
    data['tree_dbh/cb_num'] = data['tree_dbh'] / data['cb_num']
    data['boro_ct/steward'] = data['boro_ct'] / data['steward']

    data = data.drop('borocode', axis=1)

    train = data[:length]
    test  = data[length:].drop(yml['target_col'], axis=1)

    return train, test

def poor_process(train, test, yml):
    length = len(train)
    data = pd.concat([train, test])

    # 0.388を超えてるやつ
    data['created_at*st_assem'] = data['created_at'] * data['st_assem']
    data['tree_dbh*boro_ct'] = data['tree_dbh'] * data['boro_ct']
    data['cb_num*cncldist'] = data['cb_num'] * data['cncldist']
    data['created_at/borocode'] = data['created_at'] / data['borocode']
    data['created_at/cb_num'] = data['created_at'] / data['cb_num']
    data['tree_dbh/created_at'] = data['tree_dbh'] / data['created_at']
    data['tree_dbh/borocode'] = data['tree_dbh'] / data['borocode']
    data['tree_dbh/st_senate'] = data['tree_dbh'] / data['st_senate']
    data['borocode/boro_ct'] = data['borocode'] / data['boro_ct']
    data['boro_ct/tree_dbh'] = data['boro_ct'] / data['tree_dbh']
    data['boro_ct/cb_num'] = data['boro_ct'] / data['cb_num']
    data['cb_num/created_at'] = data['cb_num'] / data['created_at']
    data['cb_num/boro_ct'] = data['cb_num'] / data['boro_ct']

    # border_score_us_fairが0.325、border_score_us_poorが0.385を両方超えたやつ
    data['tree_dbh/cb_num'] = data['tree_dbh'] / data['cb_num']
    data['boro_ct/steward'] = data['boro_ct'] / data['steward']

    data = data.drop('borocode', axis=1)

    train = data[:length]
    test  = data[length:].drop(yml['target_col'], axis=1)

    return train, test