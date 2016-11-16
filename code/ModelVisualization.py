import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns


if __name__ == '__main__':

    # import data to use: this season
    elastic = pd.read_csv('data/final_EN.csv')
    elastic = elastic[elastic['home_team_win'] != -100]
    lasso5 = pd.read_csv('data/final_Lasso5.csv')
    lasso5 = lasso5[lasso5['home_team_win'] != -100]
    lasso10 = pd.read_csv('data/final_Lasso10.csv')
    lasso10 = lasso10[lasso10['home_team_win'] != -100]
    mlp = pd.read_csv('data/final_MLP.csv')
    mlp = mlp[mlp['home_team_win'] != -100]
    mlpr = pd.read_csv('data/final_MLPR.csv')
    mlpr = mlpr[mlpr['home_team_win'] != -100]
    xgbc = pd.read_csv('data/final_XGBC.csv')
    xgbc = xgbc[xgbc['home_team_win'] != -100]
    ridge = pd.read_csv('data/final_Ridge.csv')
    ridge = ridge[ridge['home_team_win'] != -100]

    # import data to use: last season
    elasticLS = pd.read_csv('data/finalLS_EN.csv')
    lasso5LS = pd.read_csv('data/finalLS_Lasso5.csv')
    lasso10LS = pd.read_csv('data/finalLS_Lasso10.csv')
    mlpLS = pd.read_csv('data/finalLS_MLP.csv')
    mlprLS = pd.read_csv('data/finalLS_MLPR.csv')
    xgbcLS = pd.read_csv('data/finalLS_XGBC.csv')
    ridgeLS = pd.read_csv('data/finalLS_Ridge.csv')

    sns.set_style("darkgrid")
    sns.set_color_codes('dark')
    sns.set(font_scale = 1.5)

    plt.figure(figsize = (18, 9))
    plt.subplot(2,1,1)
    plt.plot(elasticLS['cumulative_average'], color = 'm', label = 'Elastic Net')
    plt.plot(lasso10LS['cumulative_average'], color = 'g', label = 'Lasso Regression')
    plt.plot(ridgeLS['cumulative_average'], color = 'b', label = 'Ridge Regression')
    plt.plot(mlpLS['cumulative_average'], color = 'r', label = 'MLP Classifier')
    plt.plot(mlprLS['cumulative_average'], color = 'y', label = 'MLP Regressor')
    plt.tick_params(axis = 'x', which = 'both', bottom = 'off', top = 'off', labelbottom = 'off')
    plt.axis([-5, 1155, 0, 1])
    plt.legend(loc = 4)
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('2015-2016 Data')
    plt.subplot(2,1,2)
    plt.plot(elastic['cumulative_average'], color = 'm', label = 'Elastic Net')
    plt.plot(lasso10['cumulative_average'], color = 'g', label = 'Lasso Regression')
    plt.plot(ridge['cumulative_average'], color = 'b', label = 'Ridge Regression')
    plt.plot(mlp['cumulative_average'], color = 'r', label = 'MLP Classifier')
    plt.plot(mlpr['cumulative_average'], color = 'y', label = 'MLP Regressor')
    plt.tick_params(axis = 'x', which = 'both', bottom = 'off', top = 'off', labelbottom = 'off')
    plt.axis([-5, 1155, 0, 1])
    plt.legend(loc = 4)
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('2016-2017 Data')
    plt.savefig('CumulativeAccuracy.png')
