import pandas as pd
import json
from fun_fac import *
import numpy as np
import matplotlib.pyplot as plt
df_1 = load_and_process_data('trade_sz301596')
df_2 = load_and_process_data('snap_sz301596')
df_3 = load_and_process_data('order_sz301596')


if __name__ == "__main__":
    type1,type2,type7,type8 = gerner_win(df_1,df_2,df_3)
    # se = type2.iloc[:, 9:40].mean(axis=0)
    # plt.plot(se)
    print(type1)
    print(type2)
    print(type7)
    print(type8)