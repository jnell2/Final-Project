import pandas as pd
import numpy as np
import matplotlib.pyplot as plt






if __name__ == '__main__':

    elastic = pd.read_csv('data/final_EN.csv')
    lasso5 = pd.read_csv('data/final_Lasso5.csv')
    lasso10 = pd.read_csv('data/final_Lasso10.csv')
    MLP = pd.read_csv('data/final_MLP.csv')
    MLPR = pd.read_csv('data/final_MLPR.csv')
    XGBC = pd.read_csv('data/final_XGBC.csv')
