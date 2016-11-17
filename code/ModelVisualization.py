import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns

if __name__ == '__main__':

    # import data to use: this season
    logistic = pd.read_csv('data/final_logistic.csv')
    logistic = logistic[logistic['home_team_win'] != -100]
    lasso10 = pd.read_csv('data/final_Lasso10.csv')
    lasso10 = lasso10[lasso10['home_team_win'] != -100]
    mlp = pd.read_csv('data/final_MLP.csv')
    mlp = mlp[mlp['home_team_win'] != -100]
    mlpr = pd.read_csv('data/final_MLPR.csv')
    mlpr = mlpr[mlpr['home_team_win'] != -100]
    gbc = pd.read_csv('data/final_gbc.csv')
    gbc = gbc[gbc['home_team_win'] != -100]

    # import data to use: last season
    logisticLS = pd.read_csv('data/finalLS_logistic.csv')
    lasso10LS = pd.read_csv('data/finalLS_Lasso10.csv')
    mlpLS = pd.read_csv('data/finalLS_MLP.csv')
    mlprLS = pd.read_csv('data/finalLS_MLPR.csv')
    gbcLS = pd.read_csv('data/finalLS_gbc.csv')

    sns.set_style("darkgrid")
    sns.set_color_codes('dark')
    sns.set(font_scale = 1.5)

    plt.figure(figsize = (18, 9))
    plt.subplot(2,1,1)
    plt.plot(gbcLS['cumulative_average'], color = 'r', label = 'Gradient Boosting Classifier')
    plt.plot(lasso10LS['cumulative_average'], color = 'm', label = 'Lasso Regression')
    plt.plot(logisticLS['cumulative_average'], color = 'b', label = 'Logistic Regression')
    plt.plot(mlpLS['cumulative_average'], color = 'y', label = 'MLP Classifier')
    plt.plot(mlprLS['cumulative_average'], color = 'g', label = 'MLP Regressor')
    plt.tick_params(axis = 'x', which = 'both', bottom = 'off', top = 'off', labelbottom = 'off')
    plt.axis([-5, 1155, 0, 1])
    plt.legend(loc = 4)
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('2015-2016 Data')
    plt.subplot(2,1,2)
    plt.plot(gbc['cumulative_average'], color = 'r', label = 'Gradient Boosting Classifier')
    plt.plot(lasso10['cumulative_average'], color = 'm', label = 'Lasso Regression')
    plt.plot(logistic['cumulative_average'], color = 'b', label = 'Logistic Regression')
    plt.plot(mlp['cumulative_average'], color = 'y', label = 'MLP Classifier')
    plt.plot(mlpr['cumulative_average'], color = 'g', label = 'MLP Regressor')
    plt.tick_params(axis = 'x', which = 'both', bottom = 'off', top = 'off', labelbottom = 'off')
    plt.axis([-5, 1155, 0, 1])
    plt.legend(loc = 4)
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.title('2016-2017 Data')
    plt.savefig('images/CumulativeAccuracy.png')
