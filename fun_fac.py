import os
import pandas as pd
from itertools import zip_longest
import numpy as np
import swifter
# 详细说明文档参看飞书文档
def get_files_by_subfolder(folder_path):
    """
    遍历指定文件夹中的所有子文件夹，获取每个子文件夹中的文件。
    参数:
    folder_path (str): 要遍历的文件夹路径
    返回:
    list[list[pandas]]: 包含每个子文件夹中的文件的列表
    """
    folder_files = []
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    subfolders.sort()
    for root in subfolders[:1]:
        files_in_subfolder = [os.path.join(root,filename) for filename in os.listdir(root)]
        files_in_subfolder.sort()
        files_in_subfolder = [pd.read_json(path) for path in files_in_subfolder[:2]]
        folder_files.append(files_in_subfolder)
    return folder_files

def cal_wei_price(df):
    """
    计算加权平均价格
    参数:
    df (pandas.DataFrame): 包含'trade_price'和'trade_volume'列的DataFrame
    返回:
    float: 加权平均价格
    """
    total_sum = 0
    for i in range(df.shape[0]):
        temp_sum = df.iloc[i]['trade_price'] * df.iloc[i]['trade_volume']
        total_sum += temp_sum 
    sum_vol = sum(df[df['trade_price']!=0]['trade_volume'])
    if sum_vol == 0:
        wei_price = 0
    else:
        wei_price = total_sum / sum_vol
    return wei_price

def cal_type(series):
    cur_flag = series['flag']
    cur_price = series['price']
    cur_vol = series['volume']
    bid_1 = series['buy_delegations'][0]['price']
    bid_vol_1 = series['buy_delegations'][0]['volume']
    ask_1 = series['sell_delegations'][0]['price']
    ask_vol_1 = series['sell_delegations'][0]['volume']

    if cur_flag == 'B':
        if (cur_price > ask_1) and (cur_vol > ask_vol_1):
            return 'type1'
        elif (cur_price == ask_1) and (cur_vol > ask_vol_1):
            return 'type2'
        else :
            return 'other'
    else :
        if (cur_price < bid_1) and (cur_vol > bid_vol_1):
            return  'type7'
        elif (cur_price == bid_1) and (cur_vol > bid_vol_1):
            return 'type8'
        else:
            return 'other'

def cal_win(series,df_1):
    cur_price = series['price']
    cur_vol = series['volume']
    cur_time = series['market_time']
    weight_price = []
    for j in range(-10,21):
        k = j
        cor_time = cur_time + np.timedelta64(j, 's')
        cor_trade = df_1[df_1['timestamp'] == cor_time]
        weight_price_j = cal_wei_price(cor_trade)
        while weight_price_j == 0:
            k -= 1
            cor_time = cur_time + np.timedelta64(k,'s')
            cor_trade = df_1[df_1['timestamp'] == cor_time]
            weight_price_j = cal_wei_price(cor_trade)
        weight_price.append({j:weight_price_j})
    return weight_price
    
def gerner_win(args):
    """
    计算盈利
    参数:
    list (list): 某一只股票的三个数据信息
    df_1:trade
    df_2:snap
    df_3:order
    返回:
    """
    i,j,df_1,df_2,df_3 = args
    # df_1,df_2,df_3 = list[0],list[1],list[2]
    start_time = df_1.loc[0]['timestamp'] + np.timedelta64(910,'s')
    end_time = df_1.iloc[-1]['timestamp'] + np.timedelta64(-201,'s')
    df_3 = df_3.sort_values(by='market_time')
    order_snap_df = pd.merge_asof(df_3,df_2,
                                  left_on='market_time',right_on='market_time',
                                  direction='backward' )
    # 进行类型划分
    order_snap_df['type_agg'] = order_snap_df.apply(cal_type,axis=1)
    df_3 = order_snap_df[order_snap_df['type_agg']!='other']
    df_3 = df_3[(df_3['market_time']>start_time) & (df_3['market_time']<end_time)]
    df_3['time_win'] = df_3.swifter.apply(cal_win,axis=1,args=(df_1,))
    # for i in range(df_3.shape[0]):
    #     if df_3.iloc[i]['market_time'] < start_time or df_3.iloc[i]['market_time'] > end_time:
    #         continue
    #     current_order = df_3.iloc[i]
    #     # 获取挂单价格
    #     cur_price = current_order['price']
    #     # 获取挂单量'
    #     cur_vol = current_order['volume']
    #     # 获取时间
    #     cur_time = current_order['market_time']

    #     # 获取交易窗口数据
    #     weight_price = []
    #     for j in range(-10,21):
    #         k = j
    #         cor_time = cur_time + np.timedelta64(j, 's')
    #         cor_trade = df_1[df_1['timestamp'] == cor_time]
    #         weight_price_j = cal_wei_price(cor_trade)
    #         while weight_price_j == 0:
    #             k -= 1
    #             cor_time = cur_time + np.timedelta64(k,'s')
    #             cor_trade = df_1[df_1['timestamp'] == cor_time]
    #             weight_price_j = cal_wei_price(cor_trade)
    #         weight_price.append({j:cal_wei_price(cor_trade)})

    #     # if 'time_win' not in df_3.columns:
    #     #     df_3['time_win'] = None
    #     df_3.iloc[i,'time_win'] = weight_price

    index1 = df_3[df_3['type_agg']=='type1']['market_time']
    index2 = df_3[df_3['type_agg']=='type2']['market_time']
    index7 = df_3[df_3['type_agg']=='type7']['market_time']
    index8 = df_3[df_3['type_agg']=='type8']['market_time']

    type1_win = df_3[df_3['type_agg']=='type1']['time_win']
    type1_win.index = index1
    type2_win = df_3[df_3['type_agg']=='type2']['time_win']
    type2_win.index = index2
    type7_win = df_3[df_3['type_agg']=='type7']['time_win']
    type7_win.index = index7
    type8_win = df_3[df_3['type_agg']=='type8']['time_win']
    type8_win.index = index8

    type_1_df = pd.DataFrame(type1_win)
    type_2_df = pd.DataFrame(type2_win)
    type_7_df = pd.DataFrame(type7_win)
    type_8_df = pd.DataFrame(type8_win)

    for i in range (-10,21):
        type_1_df[str(i)]=type_1_df['time_win'].apply(lambda x: x[i+10][i])
        type_2_df[str(i)]=type_2_df['time_win'].apply(lambda x: x[i+10][i])
        type_7_df[str(i)]=type_7_df['time_win'].apply(lambda x: x[i+10][i])
        type_8_df[str(i)]=type_8_df['time_win'].apply(lambda x: x[i+10][i])

    type_1_gr = type_1_df.groupby(level=0).min()
    type_1_gr = type_1_gr.iloc[:,1:]
    type_2_gr = type_2_df.groupby(level=0).min()
    type_2_gr = type_2_gr.iloc[:,1:]
    type_7_gr = type_7_df.groupby(level=0).min()
    type_7_gr = type_7_gr.iloc[:,1:]
    type_8_gr = type_8_df.groupby(level=0).min()
    type_8_gr = type_8_gr.iloc[:,1:]

    # 进行数据标准化过程
    type_1_gr = norm(type_1_gr)
    type_2_gr = norm(type_2_gr)
    type_7_gr = norm(type_7_gr)
    type_8_gr = norm(type_8_gr)

    type_1_avg_1 = type_1_gr.mean(axis = 0)
    type_2_avg_1 = type_2_gr.mean(axis = 0)
    type_7_avg_1 = type_7_gr.mean(axis = 0)
    type_8_avg_1 = type_8_gr.mean(axis = 0)

    # 一级平均
    return (i,j,type_1_avg_1,type_2_avg_1,type_7_avg_1,type_8_avg_1)

def norm(df):
    for k in range(df.shape[0]):
        if df.iloc[k,10]== 0:
            for i in range(0,31):
                df.iloc[k,i] += 100 
        else:
            for i in range(0,10):
                df.iloc[k,i] = df.iloc[k,i] *100 / df.iloc[k,10]
            for i in range(11,31):
                df.iloc[k,i] = df.iloc[k,i] *100 / df.iloc[k,10]
            df.iloc[k,10] = 100
    return df