import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_importance(models, df, save_path, num):
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df["feature_importance"] = model.feature_importance()
        _df["column"] = df.columns
        _df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)

    order = (
        feature_importance_df.groupby("column")
        .sum()[["feature_importance"]]
        .sort_values("feature_importance", ascending=False)
        .index[:50]
    )

    fig, ax = plt.subplots(figsize=(12, max(6, len(order) * 0.25)))
    sns.boxenplot(
        data=feature_importance_df,
        x="feature_importance",
        y="column",
        order=order,
        ax=ax,
        palette="viridis",
        orient="h",
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Importance")
    ax.grid()
    fig.tight_layout()
    plt.savefig(save_path + "/visualize_importance_" + num +".png")
    
def visualize_oof_gt(oof, gt, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(oof, gt, alpha=0.5)
    gt_max = gt.max()
    ax.plot(np.arange(0, gt_max), np.arange(0, gt_max), color="red", alpha=0.5, linestyle="--")
    ax.set_xlabel("Out Of Fold")
    ax.set_ylabel("Ground Truth")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_path + "/visualize_oof_gt.png")
    
def visualize_oof_pred(train, oof, pred, save_path, num, oof_switch):
    fig, ax = plt.subplots(figsize=(8, 6))

    if oof_switch:
        ax.hist((train, oof, pred), density=True, label=['Train', 'OutOfFold', 'Test'])
    # bins = 100
    else:
        ax.hist((train, pred), density=True, label=['Train', 'Test'])
    # ax.hist(train, density=True, alpha=0.5, width=-0.3, label="Train")
    # ax.hist(pred, density=True, alpha=0.5, width=0.3, label="Test")
    # ax.hist(oof, density=True, alpha=0.5, width=0, label="OutOfFold")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_path + "/visualize_oof_pred_" + num + ".png")
