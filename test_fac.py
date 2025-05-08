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
import sys
file_paths = sys.argv[1:]

"测试"
if __name__ == "__main__":
    res1_list = []
    res2_list = []
    res7_list = []
    res8_list = []
    # with open(file_paths[0],'r') as f:
    #     data = json.load(f)
    # for record in data:
    #     record['tick_time'] = pd.to_datetime(record['tick_time'], format='%Y%m%d%H%M%S%f', errors='coerce')
    # df_1 = pd.DataFrame(data)
    df_1 = load_and_process_data(file_paths[0])
    df_2 = load_and_process_data(file_paths[1])
    df_3 = load_and_process_data(file_paths[2])
    # 只保留部分重要的列
    list_1 = ['code','tick_time','trade_price','trade_volume','trade_value']
    list_2 = ['tick_time', 'code','buy_delegations','sell_delegations']
    list_3 = ['code','flag','price', 'volume','tick_time']

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




