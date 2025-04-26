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
    # type_1_avg_ = pd.DataFrame(np.nan,index =[i for i in range(-10,21)],columns = ['mo'])
    # type_2_avg_ = pd.DataFrame(np.nan,index =[i for i in range(-10,21)],columns = ['mo'])
    # type_7_avg_ = pd.DataFrame(np.nan,index =[i for i in range(-10,21)],columns = ['mo'])
    # type_8_avg_ = pd.DataFrame(np.nan,index =[i for i in range(-10,21)],columns = ['mo'])
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
        print(res1, res2, res7, res8 )
        # tasks = []
        # for i,(row1,row2,row3) in enumerate(zip(df1_list, df2_list, df3_list)):
        #     for j,(a,b,c) in enumerate(zip(row1, row2, row3)):
        #         tasks.append((i,j,a,b,c))
        # with multiprocessing.Pool() as pool:
        #     results = pool.map(gerner_win, tasks)

        # type_1 = [[0] * len(df1[0]) for _ in range(len(df1))]
        # type_2 = [[0] * len(df1[0]) for _ in range(len(df1))]
        # type_7 = [[0] * len(df1[0]) for _ in range(len(df1))]
        # type_8 = [[0] * len(df1_list[0]) for _ in range(len(df1))]

        # for i, j, res1, res2, res7, res8 in results:
        #     type_1[i][j] = res1
        #     type_2[i][j] = res2
        #     type_7[i][j] = res7
        #     type_8[i][j] = res8

        # type_1_df = [s for sublist in type_1 for s in sublist]
        # type_2_df = [s for sublist in type_2 for s in sublist]
        # type_7_df = [s for sublist in type_7 for s in sublist]
        # type_8_df = [s for sublist in type_8 for s in sublist]
        # print(type_8_df)

        # type_1_df = pd.DataFrame(type_1_df)
        # type_2_df = pd.DataFrame(type_2_df)
        # type_7_df = pd.DataFrame(type_7_df)
        # type_8_df = pd.DataFrame(type_8_df)

        # type_1_se = type_1_df.mean(axis=0)
        # type_1_se2df = pd.DataFrame(type_1_se,columns = [str(n)])
        # type_1_avg_ = pd.concat([type_1_avg,type_1_se2df],axis=1)
                                    
        # type_2_se = type_2_df.mean(axis=0)
        # type_2_se2df = pd.DataFrame(type_2_se,columns = [str(n)])
        # type_2_avg_ = pd.concat([type_2_avg,type_2_se2df],axis=1)
                                    
        # type_7_se = type_7_df.mean(axis=0)
        # type_7_se2df = pd.DataFrame(type_1_se,columns = [str(n)])
        # type_7_avg_ = pd.concat([type_7_avg,type_7_se2df],axis=1)
                                    
                                    
    #     type_8_se = type_8_df.mean(axis=0)
    #     type_8_se2df = pd.DataFrame(type_1_se,columns = [str(n)])
    #     type_8_avg_ = pd.concat([type_8_avg,type_8_se2df],axis=1)
    #
    # print(type_8_avg_)
    
    # series_list = [type_1_se, type_2_se, type_7_se, type_8_se]
    # print(series_list)
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




