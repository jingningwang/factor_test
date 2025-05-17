import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import json

def get_files_by_subfolder(folder_path,n):
    """
    遍历指定文件夹中的所有子文件夹，获取每个子文件夹中的文件.该函数已经弃用
    input:
    folder_path: 要遍历的文件夹路径
    output:
    该文件夹中部分或者所有日期，股票数据，返回一个dataframe
    """
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    subfolders.sort()
    all_dataframes = []
    # 遍历一定的天数
    for root in subfolders[0:1]:
        files_in_subfolder = [os.path.join(root,filename) for filename in os.listdir(root)]
        files_in_subfolder.sort()
        files_in_subfolder_1 = [cut_time(pd.read_json(path)) for path in files_in_subfolder[n:n+1]]
        all_dataframes.extend(files_in_subfolder_1)
    df = pd.concat(all_dataframes,axis=0)
    return df


def load_and_process_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    df['tick_time'] = pd.to_datetime(df['tick_time'], format='%Y%m%d%H%M%S%f', errors='coerce')
    # 如果列中存在其他时间字段，批量转换
    for col in ['market_time', 'timestamp']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def cut_time(df):
    # 对传进来的数据集，去掉集合竞价阶段
    start_time = df['tick_time'].min() + np.timedelta64(910, 's')
    end_time = df['tick_time'].max() + np.timedelta64(-210, 's')
    df = df[(df['tick_time'] > start_time) & (df['tick_time'] < end_time)]
    return df

def cal_wei_price(df):
    """
    trade 数据文件
    input:
    df : 包含'trade_price'和'trade_volume'列的 DataFrame
    output:
    """
    valid_df = df[df['trade_price'] > 0]
    result = valid_df.groupby(['code', 'timestamp']).agg(
        weighted_price=('trade_value', 'sum'),
        total_volume=('trade_volume', 'sum')).reset_index()
    result['weighted_price'] = result['weighted_price'] / result['total_volume']
    return result

def cal_type(df):
    """
    计算每个挂单的类型
    input:
    order_snap_df联合df
    output:
    order_snap_df,添加了生成对应类型的列
    """
    cur_flag = df['flag']
    cur_price = df['price']
    cur_vol = df['volume']

    bid_1 = df['buy_delegations'].apply(lambda x: x[0]['price'])
    df['bid_1']=bid_1
    bid_vol_1 = df['buy_delegations'].apply(lambda x: x[0]['volume'])
    df['bid_vol_1']=bid_vol_1

    ask_1 = df['sell_delegations'].apply(lambda x: x[0]['price'])
    df['ask_1']=ask_1
    ask_vol_1 = df['sell_delegations'].apply(lambda x: x[0]['volume'])
    df['ask_vol_1']=ask_vol_1

    # 创建条件判断
    # 对于买单（cur_flag == 'B'）
    condition_b_type1 = (cur_flag == 'B') & (cur_price > ask_1) & (cur_vol > ask_vol_1)
    condition_b_type2 = (cur_flag == 'B') & (cur_price == ask_1) & (cur_vol > ask_vol_1)
    # 对于卖单（cur_flag != 'B'）
    condition_s_type7 = (cur_flag == 'S') & (cur_price < bid_1) & (cur_vol > bid_vol_1)
    condition_s_type8 = (cur_flag == 'S') & (cur_price == bid_1) & (cur_vol > bid_vol_1)

    # 使用 np.select 来向量化处理不同类型
    df['type_agg'] = np.select(
        [condition_b_type1, condition_b_type2, condition_s_type7, condition_s_type8],
        ['type1', 'type2', 'type7', 'type8'],
        default='other'  # 如果都不满足条件，返回 'other'
    )
    return df

def cal_win(row,df_1,t):
    """
    对每个类型的挂单，计算特定窗口的加权交易价
    apply 操作，时间运行状态
    input:
    series: 挂单数据
    df_1: 已经计算好教权交易价的成交数据
    t: 时间窗口平移量
    output:
    某个窗口的加权交易价
    """
    current_time = row['market_time_x']
    cor_time = current_time + np.timedelta64(t, 's')
    # 当前挂单时间，可能没有进行交易，只有撤单
    weight_price_t = df_1[df_1['timestamp'] <= cor_time].iloc[-1:]['weighted_price']
    if not weight_price_t.empty:
        return weight_price_t.iloc[0]
    else:
        return np.nan
    
def gerner_win(df_1,df_2,df_3):
    """
    input:
    df_1:trade
    df_2:snap
    df_3:order

    output:
    四种类型的交易价时间窗口
    """
    # df_3 = df_3.sort_values(by=['code','tick_time'],ascending=[True,True])
    # order_snap_df = pd.merge(df_3,df_2,on=['tick_time','code'],how='left')
    order_snap_df = pd.merge_asof(df_3.sort_values('tick_time'), df_2.sort_values('tick_time'), on='tick_time', direction='backward')

    # 对于部分数据集可能开盘出现nan
    # 但是现在使用近似匹配就不会出现nan值了
    # order_snap_df.ffill(inplace=True)
    # order_snap_df.bfill(inplace=True)
    # order_snap_df.dropna(subset=['buy_delegations'],inplace=True)
    # order_snap_df.dropna(subset=['sell_delegations'],inplace=True)

    # 部分数据集有快照字段，但没有具体的数据
    order_snap_df['buy_delegations'] = order_snap_df['buy_delegations'].apply(lambda x: x if 'price' in x[0] else np.nan)
    order_snap_df['sell_delegations'] = order_snap_df['sell_delegations'].apply(lambda x: x if 'price' in x[0] else np.nan)
    order_snap_df.dropna(subset=['buy_delegations'],inplace=True)
    order_snap_df.dropna(subset=['sell_delegations'],inplace=True)
    order_snap_df = order_snap_df.reset_index(drop=True)

    # 进行类型划分
    order_snap_df = cal_type(order_snap_df)
    # 生成窗口快照
    # 这地方一定要注意先生成窗口快照，然后再筛选类型，窗口应该是不准的, 这里生成的是不准的。
    for k in range(-10,21):
        order_snap_df[str(k) + 'ask_1'] = order_snap_df['ask_1'].shift(-k)
        order_snap_df[str(k) + 'bid_1'] = order_snap_df['bid_1'].shift(-k)

    order_snap_df = order_snap_df[order_snap_df['type_agg'] != 'other']
    order_snap_df= order_snap_df.drop_duplicates(subset='market_time_y', keep='first')
    df_3 = order_snap_df.copy()
    # 对于交易数据生成每个时刻的交易价，返回的是分组之后的 df

    df_1 = cal_wei_price(df_1)
    # 时间窗口生成
    for t in tqdm(range(-10,21)):
        if df_3.apply(cal_win, axis=1, args=(df_1,t)).empty:
            df_3.loc[:,str(t)] = pd.Series([np.nan for _ in range(df_3.shape[0])])
        else:
            df_3.loc[:,str(t)] = df_3.apply(cal_win, axis=1, args=(df_1,t))

    time_win = ['code_x','price','volume','market_time_x','market_time_y','ask_1','bid_1','ask_vol_1','bid_vol_1']
    time_win.extend([str(i) for i in range(-10,21)])
    time_win.extend([str(i)+'bid_1' for i in range(-10, 21)])
    time_win.extend([str(i)+'ask_1'for i in range(-10, 21)])
    # 在此之后采取分组处理
    type1_win = df_3[df_3['type_agg']=='type1'][time_win]
    type2_win = df_3[df_3['type_agg']=='type2'][time_win]
    type7_win = df_3[df_3['type_agg']=='type7'][time_win]
    type8_win = df_3[df_3['type_agg']=='type8'][time_win]

    # # 计算回复率窗口
    type_1_gr = return_rate_3(type1_win)
    type_2_gr = return_rate_3(type2_win)
    type_7_gr = return_rate_4(type7_win)
    type_8_gr = return_rate_4(type8_win)
    #
    cut_columns = [str(t) for t in range(-10,21)]
    cut_columns.extend(['rr_' + str(t) for t in range(1, 16)])
    type_1_gr= type_1_gr[cut_columns]
    type_2_gr= type_2_gr[cut_columns]
    type_7_gr= type_7_gr[cut_columns]
    type_8_gr= type_8_gr[cut_columns]
    # # 正则化过程
    type_1_gr = norm(type_1_gr)
    type_2_gr = norm(type_2_gr)
    type_7_gr = norm(type_7_gr)
    type_8_gr = norm(type_8_gr)

    type_1_avg_1 = type_1_gr.mean(axis = 0)
    type_2_avg_1 = type_2_gr.mean(axis = 0)
    type_7_avg_1 = type_7_gr.mean(axis = 0)
    type_8_avg_1 = type_8_gr.mean(axis = 0)

    # 一级平均
    # return type1_win, type2_win, type7_win, type8_win
    return type_1_avg_1,type_2_avg_1,type_7_avg_1,type_8_avg_1


def norm(df):
    # 先对 df 的 '0' 列为 0 的行进行处理，避免逐行处理
    df.loc[:,'0'] = np.where(df['0'] == 0, df['0'] + 100, df['0'])
    # 对于非零 '0' 列的行，将对应列的值除以 '0' 列的值，并乘以 100
    cols = [str(i) for i in range(-10, 0)] + [str(i) for i in range(1, 21)]
    df.loc[:,cols] = df.loc[:,cols].divide(df['0'], axis=0) * 100
    # 将 '0' 列的值设置为 100
    df.loc[:,'0'] = 100
    return df

def cal_cor(df):
    """
    输入:
    df:每一列对应一直股票一天的时间窗口
       每一行对应一个时刻所有股票所有日期的数据
    输出:
    cor:list,相关性检验结果
    """
    cor = []
    for t in range(0,16):
        vec_1 = df.loc[str(t)] - df.loc['-1'] 
        vec_2 = df.loc[str(t+5)] - df.loc[str(t)]
        cor.append(spearmanr(vec_1, vec_2))
    return cor

def return_rate_1(df):
    """
    输入:
    df:挂单数据,包含每个窗口的买一卖一价
    输出:
    return_rate:list,回报率列表
    """
    for k in range(1,16):
        buy = df[str(k)+'bid_1']
        sell = df['0ask_1']
        rr = sell/buy -1
        df['rr' + str(k)] = rr
    return df

def return_rate_2(df):
    """
    输入:
    df:挂单数据,包含每个窗口的买一卖一价
    输出:
    df:增加汇报率的列，增加了 15列
    """
    for k in range(1,16):
        sell = df[str(k)+'ask_1']
        buy = df['0bid_1']
        rr = sell/buy -1
        df['rr' + str(k)] = rr
    return df

def return_rate_3(df):
    for k in range(1,16):
        sell = df['0']
        buy = df[str(k)]
        rr = sell/buy -1
        df['rr_' + str(k)] = rr
    return df

def return_rate_4(df):
    for k in range(1,16):
        buy = df['0']
        sell = df[str(k)]
        rr = sell/buy -1
        df['rr_' + str(k)] = rr
    return df



