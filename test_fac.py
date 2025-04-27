import pandas as pd 
import numpy as np
import os
import json
from scipy.stats import pearsonr, spearmanr
from itertools import zip_longest
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
from tqdm import tqdm
from fun_fac import *


folder_path_1 = 'quote/trade_in_hour'
folder_path_2 = 'quote/snap_in_hour'
folder_path_3 = 'quote/order_in_hour'

"测试"
if __name__ == "__main__":
    res1_list = []
    res2_list = []
    res7_list = []
    res8_list = []
    for n in tqdm(range(0,2)):
        df_1 = get_files_by_subfolder(folder_path_1,n)
        df_2 = get_files_by_subfolder(folder_path_2,n)
        df_3 = get_files_by_subfolder(folder_path_3,n)

        list_1 = ['code','timestamp','trade_price','trade_volume']
        list_2 = ['market_time', 'code','buy_delegations','sell_delegations']
        list_3 = ['code','flag','price', 'volume','market_time']
        df_1 = df_1[list_1]
        df_2 = df_2[list_2]
        df_3 = df_3[list_3]

        res1, res2, res7, res8 = gerner_win(df_1,df_2,df_3)
        res1_list.append(res1)
        res2_list.append(res2)
        res7_list.append(res7)
        res8_list.append(res8)

    res1_df = pd.concat(res1_list)
    res2_df = pd.concat(res2_list)
    res7_df = pd.concat(res7_list)
    res8_df = pd.concat(res8_list)

    cor_1 = cal_cor(res1_df)
    cor_2 = cal_cor(res2_df)
    cor_7 = cal_cor(res7_df)
    cor_8 = cal_cor(res8_df)

    type_1_avg_2 = res1_df.mean(axis=1)
    type_2_avg_2 = res2_df.mean(axis=1)
    type_7_avg_2 = res7_df.mean(axis=1)
    type_8_avg_2 = res8_df.mean(axis=1)

    
    series_list = [type_1_avg_2, type_2_avg_2, type_7_avg_2, type_8_avg_2]
    fig,axes = plt.subplots(2,2,figsize=(15,8))
    for i,(ax,s) in enumerate(zip(axes.flat,series_list)):
        ax.plot(s.index,s.values,marker='o')
        if (i ==2) or (i==3):
            ax.set_title(f'type{i+5}')
        else:
            ax.set_title(f"type{i+1}")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('type.png')




