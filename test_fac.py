import pandas as pd 
import numpy as np
import os
import json
from scipy.stats import pearsonr, spearmanr
from itertools import zip_longest
import multiprocessing
from multiprocessing import Pool
from fun_fac import *

folder_path_1 = 'quote/trade_in_hour'
folder_path_2 = 'quote/snap_in_hour'
folder_path_3 = 'quote/order_in_hour'

df1_list = get_files_by_subfolder(folder_path_1)
df2_list = get_files_by_subfolder(folder_path_2)
df3_list = get_files_by_subfolder(folder_path_3)

if __name__ == "__main__":
    tasks = []
    for i,(row1,row2,row3) in enumerate(zip(df1_list, df2_list, df3_list)):
        for j,(a,b,c) in enumerate(zip(row1, row2, row3)):
            tasks.append((i,j,a,b,c))

    with multiprocessing.Pool() as pool:
        results = pool.map(cal_win, tasks)
    
    type_1 = [[0] * len(df1_list[0]) for _ in range(len(df1_list))]
    type_2 = [[0] * len(df1_list[0]) for _ in range(len(df1_list))]
    type_7 = [[0] * len(df1_list[0]) for _ in range(len(df1_list))]
    type_8 = [[0] * len(df1_list[0]) for _ in range(len(df1_list))]

    for i, j, res1, res2, res7, res8 in results:
        type_1[i][j] = res1
        type_2[i][j] = res2
        type_7[i][j] = res7
        type_8[i][j] = res8


