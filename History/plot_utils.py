import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def plot_dice(csv_paths,modelname):
    df = pd.read_csv(csv_paths[0])
    df_ = pd.read_csv(csv_paths[1])
    res = pd.DataFrame()
    res['epoch'] = df['Step']
    res['dice'] = df['Value']
    res['val_dice'] = df_['Value']
    # res[['epoch', 'dice', 'val_dice']].plot(
    #     x='epoch',
    #     xlabel='Epochs',
    #     ylabel='dice_coef',
    #     title='')
    fig = plt.figure(figsize=(3, 3))
    plt.plot(res["dice"], label="dice_coef")
    plt.plot(res["val_dice"], label="val_dice_coef")
    plt.plot( np.argmax(res["val_dice"]),
            np.max(res["val_dice"]),
            marker="x", color="r", label="best model")
    plt.title("Dice coef")
    plt.xlabel("Epochs")
    plt.ylabel("dice_coef")
    plt.legend()
    plt.show()
    fig.savefig(modelname)

def plot_loss(csv_paths,modelname):
    df = pd.read_csv(csv_paths[2])
    df_ = pd.read_csv(csv_paths[3])
    res = pd.DataFrame()
    res['epoch'] = df['Step']
    res['loss'] = df['Value']
    res['val_loss'] = df_['Value']

    fig = plt.figure(figsize=(3, 3))
    plt.plot(res['loss'], label="loss")
    plt.plot(res['val_loss'], label="val_loss")
    plt.plot(np.argmin(res['val_loss']),
            np.min(res['val_loss']),
            marker="x", color="r", label="best model")
    plt.title("Learning curve")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    fig.savefig(modelname)

def plot_precision(csv_paths,modelname):
    df = pd.read_csv(csv_paths[4])
    df_ = pd.read_csv(csv_paths[5])
    res = pd.DataFrame()
    res['epoch'] = df['Step']
    res['precision'] = df['Value']
    res['val_precision'] = df_['Value']

    fig = plt.figure(figsize=(3, 3))
    plt.plot(res['precision'], label="precision")
    plt.plot(res['val_precision'], label="val_precision")
    plt.plot(np.argmax(res['val_precision']),
            np.max(res['val_precision']),
            marker="x", color="r", label="best model")
    plt.title("Precision")
    plt.xlabel("Epochs")
    plt.ylabel("precision")
    plt.legend()
    plt.show()
    fig.savefig(modelname)

def plot_recall(csv_paths,modelname):
    df = pd.read_csv(csv_paths[6])
    df_ = pd.read_csv(csv_paths[7])
    res = pd.DataFrame()
    res['epoch'] = df['Step']
    res['recall'] = df['Value']
    res['val_recall'] = df_['Value']

    fig = plt.figure(figsize=(3, 3))
    plt.plot(res['recall'], label="recall")
    plt.plot(res['val_recall'], label="val_recall")
    plt.plot(np.argmax(res['val_recall']),
            np.max(res['val_recall']),
            marker="x", color="r", label="best model")
    plt.title("Recall")
    plt.xlabel("Epochs")
    plt.ylabel("recall")
    plt.legend()
    plt.show()
    fig.savefig(modelname)