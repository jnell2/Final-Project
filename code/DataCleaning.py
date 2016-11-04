import pandas as pd
import numpy as np
import WebScraping as ws

abbr_dict = {'Canadiens': 'MTL',
            'Penguins': 'PIT',
            'Oilers': 'EDM',
            'Rangers': 'NYR',
            'Wings': 'DET',
            'Blackhawks': 'CHI',
            'Wild': 'MIN',
            'Capitals': 'WSH',
            'Lightning': 'TBL',
            'Sharks': 'SJS',
            'Senators': 'OTT',
            'Blues': 'STL',
            'Flyers': 'PHI',
            'Bruins': 'BOS',
            'Ducks': 'ANA',
            'Sabres': 'BUF',
            'Devils': 'NJD',
            'Flames': 'CGY',
            'Canucks': 'VAN',
            'Panthers': 'FLA',
            'Jackets': 'CBJ',
            'Leafs': 'TOR',
            'Islanders': 'NYI',
            'Jets': 'WPG',
            'Kings': 'LAK',
            'Avalanche': 'COL',
            'Stars': 'DAL',
            'Predators': 'NSH',
            'Hurricanes': 'CAR',
            'Coyotes': 'ARI'}

def clean_dts(df_dts):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the game-by-game team summary
    '''
    df_dts['team'] = df_dts['team'].str.split().str[-1]
    df_dts.replace({'team': abbr_dict}, inplace = True)
    df_dts['date'] = df_dts['game'].str.split().str[0]
    df_dts['home_ind'] = df_dts['game'].str.split().str[1]
    df_dts['home_ind'] = df_dts['home_ind'].map({'vs': 1, '@': 0})
    df_dts['L'] = df_dts['L'] + df_dts['OTL']
    df_dts.drop(['#', 'game', 'GP', 'T', 'points'])

def clean_dp(df_dp):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the game-by-game penalty report
    '''

def clean_ds(df_ds):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the game-by-game shots report
    '''

def clean_sts(df_sts, df_sts_past):
    '''
    pass in appropriate dataframes from WebScraping.py
    will return the clean dfs for the season-by-season team summaries
    '''
    df_sts['team'] = df_dts['team'].str.split().str[-1]
    df_sts.replace({'team': abbr_dict}, inplace = True)
    df_sts.drop(['T', 'ROW'])
    df_sts_past['team'] = df_dts['team'].str.split().str[-1]
    df_sts_past.replace({'team': abbr_dict}, inplace = True)
    df_sts_past.drop(['T', 'ROW'])

def clean_ss(df_ss):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the season-by-season shot report
    '''

def clean_sgf(df_sgf):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the season-by-season goals for report
    '''

def clean_sga(df_sga):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the season-by-season goals against report
    '''

def get_clean_data():

    df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga = ws.get_data()

if __name__ == '__main__':

    # df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga = ws.get_data()
