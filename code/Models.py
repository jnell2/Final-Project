import pandas as pd
import numpy as np
from math import sqrt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

def proportion_data(df_final):
    '''
    pass in ifnal dataframe from DataCleaning that you want to transform
    will return a new dataframe with proportional data (home team stat / away team stat)
    currently, this function will only return dataset for W/L as variable trying to predict
    '''
    df= df_final.copy()
    df['spread'] = (df['home_spread'] + 1e-4)/(df['away_spread'] + df['home_spread'] + 1e-4)
    df['wins'] = (df['home_wins'] + 1e-4)/(df['away_wins'] + df['home_wins'] + 1e-4)
    df['goals'] = (df['home_goals'] + 1e-4)/(df['away_goals'] + df['home_goals'] + 1e-4)
    df['shots'] = (df['home_shots'] + 1e-4)/(df['away_shots'] + df['home_shots'] + 1e-4)
    df['PP%'] = (df['home_PP%'] + 1e-4)/(df['away_PP%'] + df['home_PP%'] + 1e-4)
    df['FOW%'] = (df['home_FOW%'] + 1e-4)/(df['away_FOW%'] + df['home_FOW%'] + 1e-4)
    df['PIM'] = (df['home_PIM'] + 1e-4)/(df['away_PIM'] + df['home_PIM'] + 1e-4)
    df['hits'] = (df['home_hits'] + 1e-4)/(df['away_hits'] + df['home_hits'] + 1e-4)
    df['blocked'] = (df['home_blocked'] + 1e-4)/(df['away_blocked'] + df['home_blocked'] + 1e-4)
    df['giveaways'] = (df['home_giveaways'] + 1e-4)/(df['away_giveaways'] + df['home_giveaways'] + 1e-4)
    df['takeaways'] = (df['home_takeaways'] + 1e-4)/(df['away_takeaways'] + df['home_takeaways'] + 1e-4)
    df['save%'] = (df['home_save%'] + 1e-4)/(df['away_save%'] + df['home_save%'] + 1e-4)
    df['shot%'] = (df['home_shot%'] + 1e-4)/(df['away_shot%'] + df['home_shot%'] + 1e-4)
    df['PDO'] = (df['home_PDO'] + 1e-4)/(df['away_PDO'] + df['home_PDO'] + 1e-4)
    df['corsi'] = (df['home_corsi'] + 1e-4)/(df['away_corsi'] + df['home_corsi'] + 1e-4)
    df.drop(['home_spread', 'home_wins', 'home_goals', 'home_shots', 'home_PP%', \
    'home_FOW%', 'home_PIM', 'home_hits', 'home_blocked', 'home_giveaways', \
    'home_takeaways', 'home_save%', 'home_shot%', 'home_PDO', 'home_corsi', \
    'away_spread', 'away_wins', 'away_goals', 'away_shots', 'away_PP%', \
    'away_FOW%', 'away_PIM', 'away_hits', 'away_blocked', 'away_giveaways', \
    'away_takeaways', 'away_save%', 'away_shot%', 'away_PDO', 'away_corsi'], \
    axis = 1, inplace = True)

    return df

def make_train_test(df_final):
    '''
    import final dataframe from DataCleaning that you want to TTS
    will return two sets of train_test_split
    1) using home team W/L as variable trying to predict
    2) using spread as variable trying to predict
    '''
    df = df_final.copy()
    y1 = df.pop('home_team_win')
    # variable 1 to predict
    y2 = df.pop('spread')
    # variable 2 to predict
    df.drop(['home_team', 'away_team', 'date'], axis = 1, inplace = True)
    # we don't want any categorical variables in the model
    X = df.values

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y1, test_size = 0.3, random_state = 2)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size = 0.3, random_state = 2)

    return X_train1, X_test1, y_train1, y_test1, X_train2, X_test2, y_train2, y_test2

def drop_variables(df_final):
    '''
    drop variables to test accuracies
    '''
    df_final.drop(['home_shots', 'away_shots', 'home_shot%', 'away_shot%', 'home_blocked', 'away_blocked'], axis = 1, inplace = True)

    return df_final

def logistic(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return accuracy, y predicted values, and rmse
    only use this for predicting home team W/L (variable 1 listed below)
    '''
    accuracies = []
    precisions = []
    recalls = []
    model = LogisticRegression()
    LR_model = model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    y_true = y_test
    accuracies.append(accuracy_score(y_true, y_predict))
    accuracy = np.average(accuracies)

    return accuracy, y_predict

def lasso(X_train, X_test, y_train, y_test2):
    '''
    pass in the 4 TTS outputs, y_test from W/L TTS, not spread TTS
    will return y predicted values, accuracy
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    model = Lasso(alpha = 0.00001, fit_intercept = False, normalize = True)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_pred = map(lambda x: 1 if x > 0 else 0, model_pred)
    model_accuracy = accuracy_score(y_test2, model_pred)

    return model_pred, model_accuracy

def ridge(X_train, X_test, y_train, y_test2):
    '''
    pass in the 4 TTS outputs, y_test from W/L TTS, not spread TTS
    will return y predicted values, accuracy
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    model = Ridge(alpha = 0.00001, fit_intercept = False, normalize = True)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_pred = map(lambda x: 1 if x > 0 else 0, model_pred)
    model_accuracy = accuracy_score(y_test2, model_pred)

    return model_pred, model_accuracy

def elastic(X_train, X_test, y_train, y_test2):
    '''
    pass in the 4 TTS outputs, y_test from W/L TTS, not spread TTS
    will return y predicted values, rmse, r2
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    model = ElasticNet(alpha = 0.00001, fit_intercept = False, normalize = True)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_pred = map(lambda x: 1 if x > 0 else 0, model_pred)
    model_accuracy = accuracy_score(y_test2, model_pred)

    return model_pred, model_accuracy

def linear(X_train, X_test, y_train, y_test2):
    '''
    pass in the 4 TTS outputs, y_test from W/L TTS, not spread TTS
    will return y predicted values, accuracy
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    model = LinearRegression(fit_intercept = False, normalize = True, n_jobs = -1)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_pred = map(lambda x: 1 if x > 0 else 0, model_pred)
    model_accuracy = accuracy_score(y_test2, model_pred)

    return model_pred, model_accuracy

def sgd(X_train, X_test, y_train, y_test2):
    '''
    pass in the 4 TTS outputs, y_test from W/L TTS, not spread TTS
    will return y predicted values, accuracy
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    model = SGDRegressor(n_iter = 100)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_test)
    model_pred = map(lambda x: 1 if x > 0 else 0, model_pred)
    model_accuracy = accuracy_score(y_test2, model_pred)

    return model_pred, model_accuracy

def random_forest_classifier(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS outputs
    will return accuracy, y predicted values, and features
    only use this for predicting home team W/L (variable 1 listed below)
    '''
    rf = RandomForestClassifier(random_state = 2)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = rf.score(X_test, y_test)
    features = rf.feature_importances_

    return accuracy, y_pred, features

def random_forest_regressor(X_train, X_test, y_train, y_test2):
    '''
    pass in the 4 TTS outputs, y_test from W/L TTS, not spread TTS
    will return y predicted values, accuracy, and features
    only use this for predicting spread, variable 2 listed below (in relation to home team)
    '''
    rf = RandomForestRegressor(random_state = 2)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred = map(lambda x: 1 if x > 0 else 0, y_pred)
    features = rf.feature_importances_
    accuracy = accuracy_score(y_test2, y_pred)

    return y_pred, accuracy, features

def xgboost(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS
    will return predictions and accuracy
    '''
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)

    return predictions, accuracy

def xgboost_reg(X_train, X_test, y_train, y_test2):
    '''
    pass in the 4 TTS outputs, y_test from W/L TTS, not spread TTS
    will return predictions and accuracy
    '''
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = map(lambda x: 1 if x > 0 else 0, y_pred)
    accuracy = accuracy_score(y_test2, y_pred)

    return y_pred, accuracy

def mlp(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS
    will return predictions and accuracy
    '''
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, hidden_layer_sizes = (5,2), random_state = 2)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return predictions, accuracy

def gbc(X_train, X_test, y_train, y_test):
    '''
    pass in the 4 TTS
    will return predictions and accuracy
    '''
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return y_pred, accuracy

if __name__ == '__main__':

    # this gets the most recent final csv files
    # if you want up-to-date information, re-run DataCleaning.py
    df_final2 = pd.read_csv('data/final2.csv')
    df_final2.drop(['Unnamed: 0'], axis = 1, inplace = True)
    df_final5 = pd.read_csv('data/final5.csv')
    df_final5.drop(['Unnamed: 0'], axis = 1, inplace = True)
    df_final10 = pd.read_csv('data/final10.csv')
    df_final10.drop(['Unnamed: 0'], axis = 1, inplace = True)
    df_final15 = pd.read_csv('data/final15.csv')
    df_final15.drop(['Unnamed: 0'], axis = 1, inplace = True)

    # making new dataframes for 10, 15 games with proportion data
    # only doing this for 5, 10, and 15 games because those seem to perform the best
    df_final5_prop = proportion_data(df_final5)
    df_final10_prop = proportion_data(df_final10)
    df_final15_prop = proportion_data(df_final15)

    # drop variables
    # df_final2 = drop_variables(df_final2)
    # df_final5 = drop_variables(df_final5)
    # df_final10 = drop_variables(df_final10)
    # df_final15 = drop_variables(df_final15)

    # the goal is to predict 2 variables:
    # 1) home team W/L
    # 2) spread

    # get train_test_split for 2 games
    Xtr12, Xte12, ytr12, yte12, Xtr22, Xte22, ytr22, yte22 = make_train_test(df_final2)
    # get train_test_split for 5 games
    Xtr15, Xte15, ytr15, yte15, Xtr25, Xte25, ytr25, yte25 = make_train_test(df_final5)
    # get train_test_split for 10 games
    Xtr110, Xte110, ytr110, yte110, Xtr210, Xte210, ytr210, yte210 = make_train_test(df_final10)
    # get train_test_split for 15 games
    Xtr115, Xte115, ytr115, yte115, Xtr215, Xte215, ytr215, yte215 = make_train_test(df_final15)

    # get train_test_split for 5 games, proportion_data
    X1_train5, X1_test5, y1_train5, y1_test5, X2_train5, X2_test5, y2_train5, y2_test5 = make_train_test(df_final5_prop)
    # get train_test_split for 10 games, proportion_data
    X1_train10, X1_test10, y1_train10, y1_test10, X2_train10, X2_test10, y2_train10, y2_test10 = make_train_test(df_final10_prop)
    # get train_test_split for 15 games, proportion_data
    X1_train15, X1_test15, y1_train15, y1_test15, X2_train15, X2_test15, y2_train15, y2_test15 = make_train_test(df_final15_prop)

    # Logistic Regression results
    LogReg2_accuracy, LogReg2_predict = logistic(Xtr12, Xte12, ytr12, yte12)
    LogReg5_accuracy, LogReg5_predict = logistic(Xtr15, Xte15, ytr15, yte15)
    LogReg10_accuracy, LogReg10_predict = logistic(Xtr110, Xte110, ytr110, yte110)
    LogReg15_accuracy, LogReg15_predict = logistic(Xtr115, Xte115, ytr115, yte115)
    LogReg5pred_accuracy, LogReg5pred_predict = logistic(X1_train5, X1_test5, y1_train5, y1_test5)
    LogReg10pred_accuracy, LogReg10pred_predict = logistic(X1_train10, X1_test10, y1_train10, y1_test10)
    LogReg15pred_accuracy, LogReg15pred_predict = logistic(X1_train15, X1_test15, y1_train15, y1_test15)

    # Linear Regression results
    # Lasso
    Lasso2_pred, Lasso2_accuracy = lasso(Xtr22, Xte22, ytr22, yte12)
    Lasso5_pred, Lasso5_accuracy = lasso(Xtr25, Xte25, ytr25, yte15)
    Lasso10_pred, Lasso10_accuracy = lasso(Xtr210, Xte210, ytr210, yte110)
    Lasso15_pred, Lasso15_accuracy = lasso(Xtr215, Xte215, ytr215, yte115)
    Lasso5pred_pred, Lasso5pred_accuracy = lasso(X2_train5, X2_test5, y2_train5, y1_test5)
    Lasso10pred_pred, Lasso10pred_accuracy = lasso(X2_train10, X2_test10, y2_train10, y1_test10)
    Lasso15pred_pred, Lasso15pred_accuracy = lasso(X2_train15, X2_test15, y2_train15, y1_test15)
    # Ridge
    Ridge2_pred, Ridge2_accuracy = ridge(Xtr22, Xte22, ytr22, yte12)
    Ridge5_pred, Ridge5_accuracy = ridge(Xtr25, Xte25, ytr25, yte15)
    Ridge10_pred, Ridge10_accuracy = ridge(Xtr210, Xte210, ytr210, yte110)
    Ridge15_pred, Ridge15_accuracy = ridge(Xtr215, Xte215, ytr215, yte115)
    Ridge5pred_pred, Ridge5pred_accuracy = ridge(X2_train5, X2_test5, y2_train5, y1_test5)
    Ridge10pred_pred, Ridge10pred_accuracy = ridge(X2_train10, X2_test10, y2_train10, y1_test10)
    Ridge15pred_pred, Ridge15pred_accuracy = ridge(X2_train15, X2_test15, y2_train15, y1_test15)
    # Elastic Net
    EN2_pred, EN2_accuracy = elastic(Xtr22, Xte22, ytr22, yte12)
    EN5_pred, EN5_accuracy = elastic(Xtr25, Xte25, ytr25, yte15)
    EN10_pred, EN10_accuracy = elastic(Xtr210, Xte210, ytr210, yte110)
    EN15_pred, EN15_accuracy = elastic(Xtr215, Xte215, ytr215, yte115)
    EN5pred_pred, EN5pred_accuracy = elastic(X2_train5, X2_test5, y2_train5, y1_test5)
    EN10pred_pred, EN10pred_accuracy = elastic(X2_train10, X2_test10, y2_train10, y1_test10)
    EN15pred_pred, EN15pred_accuracy = elastic(X2_train15, X2_test15, y2_train15, y1_test15)
    # Linear Regression
    LR2_pred, LR2_accuracy = linear(Xtr22, Xte22, ytr22, yte12)
    LR5_pred, LR5_accuracy = linear(Xtr25, Xte25, ytr25, yte15)
    LR10_pred, LR10_accuracy = linear(Xtr210, Xte210, ytr210, yte110)
    LR15_pred, LR15_accuracy = linear(Xtr215, Xte215, ytr215, yte115)
    LR5pred_pred, LR5pred_accuracy = linear(X2_train5, X2_test5, y2_train5, y1_test5)
    LR10pred_pred, LR10pred_accuracy = linear(X2_train10, X2_test10, y2_train10, y1_test10)
    LR15pred_pred, LR15pred_accuracy = linear(X2_train15, X2_test15, y2_train15, y1_test15)
    # SGD Regressor
    SGD2_pred, SGD2_accuracy = sgd(Xtr22, Xte22, ytr22, yte12)
    SGD5_pred, SGD5_accuracy = sgd(Xtr25, Xte25, ytr25, yte15)
    SGD10_pred, SGD10_accuracy = sgd(Xtr210, Xte210, ytr210, yte110)
    SGD15_pred, SGD15_accuracy = sgd(Xtr215, Xte215, ytr215, yte115)
    SGD5pred_pred, SGD5pred_accuracy = sgd(X2_train5, X2_test5, y2_train5, y1_test5)
    SGD10pred_pred, SGD10pred_accuracy = sgd(X2_train10, X2_test10, y2_train10, y1_test10)
    SGD15pred_pred, SGD15pred_accuracy = sgd(X2_train15, X2_test15, y2_train15, y1_test15)

    # Random Forest Classifier results
    RFC2_accuracy, RFC2_pred, RFC2_features = random_forest_classifier(Xtr12, Xte12, ytr12, yte12)
    RFC5_accuracy, RFC5_pred, RFC5_features = random_forest_classifier(Xtr15, Xte15, ytr15, yte15)
    RFC10_accuracy, RFC10_pred, RFC10_features = random_forest_classifier(Xtr110, Xte110, ytr110, yte110)
    RFC15_accuracy, RFC15_pred, RFC15_features = random_forest_classifier(Xtr115, Xte115, ytr115, yte115)
    RFC5pred_accuracy, RFC5pred_pred, RFC5pred_features = random_forest_classifier(X1_train5, X1_test5, y1_train5, y1_test5)
    RFC10pred_accuracy, RFC10pred_pred, RFC10pred_features = random_forest_classifier(X1_train10, X1_test10, y1_train10, y1_test10)
    RFC15pred_accuracy, RFC15pred_pred, RFC15pred_features = random_forest_classifier(X1_train15, X1_test15, y1_train15, y1_test15)

    # Random Forest Regressor results
    RFR2_pred, RFR2_accuracy, RFR2_features = random_forest_regressor(Xtr22, Xte22, ytr22, yte12)
    RFR5_pred, RFR5_accuracy, RFR5_features = random_forest_regressor(Xtr25, Xte25, ytr25, yte15)
    RFR10_pred, RFR10_accuracy, RFR10_features = random_forest_regressor(Xtr210, Xte210, ytr210, yte110)
    RFR15_pred, RFR15_accuracy, RFR15_features = random_forest_regressor(Xtr215, Xte215, ytr215, yte115)
    RFR5pred_pred, RFR5pred_accuracy, RFR5pred_features = random_forest_regressor(X2_train5, X2_test5, y2_train5, y1_test5)
    RFR10pred_pred, RFR10pred_accuracy, RFR10pred_features = random_forest_regressor(X2_train10, X2_test10, y2_train10, y1_test10)
    RFR15pred_pred, RFR15pred_accuracy, RFR15pred_features = random_forest_regressor(X2_train15, X2_test15, y2_train15, y1_test15)

    # XG Boost results
    XGB2_preds, XGB2_accuracy = xgboost(Xtr12, Xte12, ytr12, yte12)
    XGB5_preds, XGB5_accuracy = xgboost(Xtr15, Xte15, ytr15, yte15)
    XGB10_preds, XGB10_accuracy = xgboost(Xtr110, Xte110, ytr110, yte110)
    XGB15_preds, XGB15_accuracy = xgboost(Xtr115, Xte115, ytr115, yte115)
    XGB5pred_preds, XGB5pred_accuracy = xgboost(X1_train5, X1_test5, y1_train5, y1_test5)
    XGB10pred_preds, XGB10pred_accuracy = xgboost(X1_train10, X1_test10, y1_train10, y1_test10)
    XGB15pred_preds, XGB15pred_accuracy = xgboost(X1_train15, X1_test15, y1_train15, y1_test15)

    # XG Boost Regressor results
    XGBr2_preds, XGBr2_accuracy = xgboost_reg(Xtr22, Xte22, ytr22, yte12)
    XGBr5_preds, XGBr5_accuracy = xgboost_reg(Xtr25, Xte25, ytr25, yte15)
    XGBr10_preds, XGBr10_accuracy = xgboost_reg(Xtr210, Xte210, ytr210, yte110)
    XGBr15_preds, XGBr15_accuracy = xgboost_reg(Xtr215, Xte215, ytr215, yte115)
    XGBr5pred_preds, XGBr5pred_accuracy = xgboost_reg(X2_train5, X2_test5, y2_train5, y1_test5)
    XGBr10pred_preds, XGBr10pred_accuracy = xgboost_reg(X2_train10, X2_test10, y2_train10, y1_test10)
    XGBr15pred_preds, XGBr15pred_accuracy = xgboost_reg(X2_train15, X2_test15, y2_train15, y1_test15)

    # MLP results
    MLP2_preds, MLP2_accuracy = mlp(Xtr12, Xte12, ytr12, yte12)
    MLP5_preds, MLP5_accuracy = mlp(Xtr15, Xte15, ytr15, yte15)
    MLP10_preds, MLP10_accuracy = mlp(Xtr110, Xte110, ytr110, yte110)
    MLP15_preds, MLP15_accuracy = mlp(Xtr115, Xte115, ytr115, yte115)
    MLP5pred_preds, MLP5pred_accuracy = mlp(X1_train5, X1_test5, y1_train5, y1_test5)
    MLP10pred_preds, MLP10pred_accuracy = mlp(X1_train10, X1_test10, y1_train10, y1_test10)
    MLP15pred_preds, MLP15pred_accuracy = mlp(X1_train15, X1_test15, y1_train15, y1_test15)

    # Gradient Boosting Classifier results
    GBC2_preds, GBC2_accuracy = gbc(Xtr12, Xte12, ytr12, yte12)
    GBC5_preds, GBC5_accuracy = gbc(Xtr15, Xte15, ytr15, yte15)
    GBC10_preds, GBC10_accuracy = gbc(Xtr110, Xte110, ytr110, yte110)
    GBC15_preds, GBC15_accuracy = gbc(Xtr115, Xte115, ytr115, yte115)
    GBC5pred_preds, GBC5pred_accuracy = gbc(X1_train5, X1_test5, y1_train5, y1_test5)
    GBC10pred_preds, GBC10pred_accuracy = gbc(X1_train10, X1_test10, y1_train10, y1_test10)
    GBC15pred_preds, GBC15pred_accuracy = gbc(X1_train15, X1_test15, y1_train15, y1_test15)

    print 'LogReg15_accuracy'
    print LogReg15_accuracy
    print 'SGD2_accuracy'
    print SGD2_accuracy
    print 'RFR10_accuracy'
    print RFR10_accuracy
    print 'LogReg10pred_accuracy'
    print LogReg10pred_accuracy
    print 'LogReg15pred_accuracy'
    print LogReg10pred_accuracy
    print 'Lasso10_accuracy'
    print Lasso10_accuracy
    print 'Lasso15_accuracy'
    print Lasso15_accuracy
    print 'Ridge10_accuracy'
    print Ridge10_accuracy
    print 'Ridge15_accuracy'
    print Ridge15_accuracy
    print 'LR10_accuracy'
    print LR10_accuracy
    print 'LR15_accuracy'
    print LR15_accuracy
    print 'RFR10pred_accuracy'
    print RFR10pred_accuracy
    print 'XGBr10_accuracy'
    print XGBr10_accuracy
    print 'MLP5pred_accuracy'
    print MLP5pred_accuracy
