import pandas as pd 
import numpy as np
import os
import json
from scipy.stats import pearsonr, spearmanr
from itertools import zip_longest
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt
# from setuptools.sandbox import save_path
from tqdm import tqdm
from fun_fac import *
import sys
file_paths = sys.argv[1:]

# folder_path_1 = 'quote/trade_in_hour'
# folder_path_2 = 'quote/snap_in_hour'
# folder_path_3 = 'quote/order_in_hour'

"测试"
if __name__ == "__main__":
    res1_list = []
    res2_list = []
    res7_list = []
    res8_list = []
    # 两只股票
    df_1 = pd.read_json(file_paths[0])
    df_2 = pd.read_json(file_paths[1])
    df_3 = pd.read_json(file_paths[2])
    # 只保留部分重要的列
    list_1 = ['code','timestamp','trade_price','trade_volume','trade_value']
    list_2 = ['market_time', 'code','buy_delegations','sell_delegations']
    list_3 = ['code','flag','price', 'volume','market_time']
    df_1 = df_1[list_1]
    df_2 = df_2[list_2]
    df_3 = df_3[list_3]
    # 去除集合竞价阶段
    df_1 = cut_time(df_1)
    df_2 = cut_time(df_2)
    df_3 = cut_time(df_3)
    res1, res2, res7, res8 = gerner_win(df_1,df_2,df_3)
    res1_list.append(res1)
    res2_list.append(res2)
    res7_list.append(res7)
    res8_list.append(res8)
    res1_df = pd.concat(res1_list,axis=1)
    res2_df = pd.concat(res2_list,axis=1)
    res7_df = pd.concat(res7_list,axis=1)
    res8_df = pd.concat(res8_list,axis=1)

    type_1_avg_2 = res1_df.mean(axis=1)
    type_2_avg_2 = res2_df.mean(axis=1)
    type_7_avg_2 = res7_df.mean(axis=1)
    type_8_avg_2 = res8_df.mean(axis=1)

    result = pd.concat([type_1_avg_2,type_2_avg_2,type_7_avg_2,type_8_avg_2],axis=1)
    name = "_".join(file_paths[0].split(os.sep)[-2:])
    save_path = "/result"
    final_path = os.path.join(save_path, name)
    result.to_json(f"{final_path}")

    # series_list = [type_1_avg_2[:31], type_2_avg_2[:31], type_7_avg_2[:31], type_8_avg_2[:31]]
    # fig,axes = plt.subplots(2,2,figsize=(15,8))
    # for i,(ax,s) in enumerate(zip(axes.flat,series_list)):
    #     ax.plot(s.index,s.values,marker='o')
    #     if (i ==2) or (i==3):
    #         ax.set_title(f'type{i+5}')
    #     else:
    #         ax.set_title(f"type{i+1}")
    #     ax.grid(True)
    # plt.tight_layout()
    # plt.savefig('type.png')




