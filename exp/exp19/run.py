import pandas as pd
import numpy as np
import yaml

from process import process, under_sampling, submit, fair_process, poor_process
from model import LIGHTGBM
from visualize import visualize_importance, visualize_oof_gt, visualize_oof_pred

all_pred_us_fair = []
all_pred_us_poor = []
all_score_us_fair = []
all_score_us_poor = []
all_oof_us_fair = []
all_oof_us_poor = []
pi_fair = {'zero': 0, 'one': 0, 'twe': 0}
pi_poor = {'zero': 0, 'one': 0, 'twe': 0}

# params
with open(R'params.yaml') as file:
    yml = yaml.safe_load(file)


for i in range(yml['ensemble_num']):
    yml['params']['seed'] = i
    if i >= yml['target_switch_num']:
        yml['target_encoding_type'] = 'cat'
    # process
    train, test = process(yml)


    fair_train, fair_test = fair_process(train, test, yml)
    # under sampling
    if i == 0:
        for c in list(pi_fair):
            pi_fair[c] = int(train['health'].value_counts()[0] * yml['run_params']['model1_under_sampling'][c])
    train_under_sampling_fair, y_resampled_fair = under_sampling(fair_train, pi_fair['zero'], pi_fair['one'], pi_fair['twe'])
    # inference
    pred_us_fair, oof_us_fair, score_us_fair, logs, models_us_fair = LIGHTGBM(train_under_sampling_fair, fair_test, yml)


    poor_train, poor_test = poor_process(train, test, yml)
    # under sampling
    if i == 0:
        for c in list(pi_poor):
            pi_poor[c] = int(train['health'].value_counts()[2] * yml['run_params']['model2_under_sampling'][c])
    train_under_sampling_poor, y_resampled_poor = under_sampling(poor_train, pi_poor['zero'], pi_poor['one'], pi_poor['twe'])
    # inference
    pred_us_poor, oof_us_poor, score_us_poor, logs, models_us_poor = LIGHTGBM(train_under_sampling_poor, poor_test, yml)
    
    all_pred_us_fair.append(pred_us_fair)
    all_pred_us_poor.append(pred_us_poor)
    all_score_us_fair.append(score_us_fair)
    all_score_us_poor.append(score_us_poor)
    all_oof_us_fair.append(oof_us_fair)
    all_oof_us_poor.append(oof_us_poor)


# とりあえず出力は加重平均
pred = np.argmax(np.average(all_pred_us_fair, axis=0) + np.average(all_pred_us_poor, axis=0), axis=1)
score_us_fair = np.average(all_score_us_fair)
score_us_poor = np.average(all_score_us_poor)
# submit
save_path = submit(pred, f'{round(score_us_fair, 5)}_{round(score_us_poor, 5)}', logs, yml)

# model1 visualize
type = 'model1'
# visualize_importance(models_us_fair, fair_test, save_path, type)
visualize_oof_pred(train[yml["target_col"]], np.argmax(np.average(all_oof_us_fair, axis=0), axis=1), np.argmax(np.average(all_pred_us_fair, axis=0), axis=1), save_path, type, True)
# model2 visualize
type = 'model2'
# visualize_importance(models_us_poor, poor_test, save_path, type)
visualize_oof_pred(train[yml["target_col"]], np.argmax(np.average(all_oof_us_poor, axis=0), axis=1), np.argmax(np.average(all_pred_us_poor, axis=0), axis=1), save_path, type, True)
# ensemble visualize
type = 'ensemble'
visualize_oof_pred(train[yml["target_col"]], all_oof_us_poor, pred, save_path, type, False)

# oofとpredのデータを保存
np.save(save_path + '/oof_us_fair.npy', all_oof_us_fair)
np.save(save_path + '/oof_us_poor.npy', all_oof_us_poor)
np.save(save_path + '/pred_us_fair.npy', all_pred_us_fair)
np.save(save_path + '/pred_us_poor.npy', all_pred_us_poor)
np.save(save_path + '/score_us_fair.npy', all_score_us_fair)
np.save(save_path + '/score_us_poor.npy', all_score_us_poor)
# y_resampled_fair = y_resampled_fair.to_numpy()
# np.save(save_path + '/y_resampled_fair.npy', y_resampled_fair)
# y_resampled_poor = y_resampled_poor.to_numpy()
# np.save(save_path + '/y_resampled_poor.npy', y_resampled_poor)