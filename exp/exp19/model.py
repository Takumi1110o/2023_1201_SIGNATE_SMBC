import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score

# https://blog.amedama.jp/entry/lightgbm-custom-metric
def lgb_f1_score(preads, data):
    y_true = data.get_label() # 正解ラベル
    N_LABELS = 3  # ラベルの数
    # 最尤と判断したクラスを選ぶ
    y_pred = np.argmax(preads, axis=1)
    score = f1_score(y_true, y_pred, average='macro')
    return 'custom', score, True

def LIGHTGBM(train, test, yml):
    score = 0
    logs = []
    models = []
    target = train[yml["target_col"]]
    features = train.drop(columns=[yml["target_col"]]).columns
    categorical_feature = train.select_dtypes(include='category').columns.tolist()
    # categorical_feature = ['region', 'manufacturer', 'condition', 'fuel', 'title_status', 'transmission', 'size', 'type', 'paint_color', 'state']
    kfold = StratifiedKFold(n_splits=yml["n_splits"], shuffle=True, random_state = yml["params"]["seed"])

    oof = np.zeros((len(train), yml['params']['num_class']), dtype=np.float64)
    # preds = pd.DataFrame(columns=[f"Fold_{i}" for i in range(yml["n_splits"])])
    preds = np.zeros((len(test), yml['params']['num_class']))

    for i, (trn_idx, val_idx) in enumerate(kfold.split(train, train['health'])):
        x_train, x_valid = train.iloc[trn_idx], train.iloc[val_idx]
        y_train, y_valid = target.iloc[trn_idx], target.iloc[val_idx]

        lgb_train = lgb.Dataset(x_train.drop(yml["target_col"], axis=1), y_train)
        lgb_valid = lgb.Dataset(x_valid.drop(yml["target_col"], axis=1), y_valid)

        history = {}
        model = lgb.train(
                params = yml["params"],
                train_set = lgb_train,
                # categorical_feature = categorical_feature,
                num_boost_round = yml["train_params"]["num_boost_round"],
                valid_sets = [lgb_train, lgb_valid],
                feval=lgb_f1_score,
                callbacks = [
                    lgb.callback.record_evaluation(history),
                    lgb.early_stopping(
                        stopping_rounds=yml["train_params"]["early_stopping_rounds"],
                        verbose=True),
                    lgb.log_evaluation(yml["train_params"]["verbose_eval"])
                ]
                )

        # predict
        oof[val_idx] = model.predict(x_valid.drop(yml["target_col"], axis=1))
        # oof[val_idx] = np.argmax(model.predict(x_valid.drop(yml["target_col"], axis=1)), axis=1)
        preds += model.predict(test)
        # preds[f"Fold_{i}"] = model.predict(test)
        
        # save 
        score += history["valid_1"][yml['params']['metric']][-1]
        logs.append(history)
        models.append(model)

    # mean
    score = score / yml["n_splits"]
    preds = preds / yml['n_splits']
    # preds = np.argmax(preds, axis=1)
    
    print("*"*30 + " finished " + "*"*30)
    print(score)
    
    return preds, oof, score, logs, models