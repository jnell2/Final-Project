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

def clean_dts(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the game-by-game team summary
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['date'] = df['game'].str.split().str[0]
    df['home_ind'] = df['game'].str.split().str[1]
    df['home_ind'] = df['home_ind'].map({'vs': 1, '@': 0})
    df['L'] = df['L'] + df['OTL']
    df['W'] = df['W'].astype(int)
    df['L'] = df['L'].astype(int)
    df['GF'] = df['GF'].astype(int)
    df['GA'] = df['GA'].astype(int)
    df['SF'] = df['SF'].astype(int)
    df['SA'] = df['SA'].astype(int)
    df['PP%'] = df['PP%'].astype(float)
    df['PK%'] = df['PK%'].astype(float)
    df['FOW%'] = df['FOW%'].astype(float)
    df.drop(['#', 'game', 'GP', 'T', 'OTL', 'points', 'PPG', 'PP', 'timesSH', 'PPGA', 'FOW', 'FOL'], axis=1, inplace=True)
    df = df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    idx = df.groupby('team')['date'].nsmallest(10).index.get_level_values(1)
    df['last10'] = np.where(df.index.isin(idx),1,0)
    idx2 = df.groupby('team')['date'].nsmallest(5).index.get_level_values(1)
    df['last5'] = np.where(df.index.isin(idx2),1,0)
    idx3 = df.groupby('team')['date'].nsmallest(2).index.get_level_values(1)
    df['last2'] = np.where(df.index.isin(idx3),1,0)
    return df

def clean_dp(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the game-by-game penalty report
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['date'] = df['game'].str.split().str[0]
    df['PIM'] = df['PIM'].astype(int)
    df['penalties'] = df['penalties'].astype(int)
    df['date'] = pd.to_datetime(df['date'])
    df.drop(['#', 'game', 'GP', 'W', 'L', 'T', 'OTL', 'points', 'minor', 'major', 'misconduct', 'game_misconduct', 'match'], axis=1, inplace=True)
    return df

def clean_ds(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the game-by-game shots report
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['date'] = df['game'].str.split().str[0]
    df['hits'] = df['hits'].astype(int)
    df['blocked_shots'] = df['blocked_shots'].astype(int)
    df['giveaways'] = df['giveaways'].astype(int)
    df['takeaways'] = df['takeaways'].astype(int)
    df['save%'] = df['save%'].astype(float)
    df['date'] = pd.to_datetime(df['date'])
    df.drop(['#', 'game', 'GP', 'W', 'L', 'T', 'OTL', 'points', 'FOW', 'FOL', 'FO', 'FOW%', 'SF', 'GF'], axis=1, inplace=True)
    return df

def clean_sts(df, df2):
    '''
    pass in appropriate dataframes from WebScraping.py
    will return the clean dfs for the season-by-season team summaries
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['L'] = df['L'] + df['OTL']
    df['rank'] = df['rank'].astype(int)
    df['GP'] = df['GP'].astype(int)
    df['W'] = df['W'].astype(int)
    df['L'] = df['L'].astype(int)
    df['point%'] = df['point%'].astype(float)
    df['GF'] = df['GF'].astype(int)
    df['GA'] = df['GA'].astype(int)
    df['GF/GP'] = df['GF/GP'].astype(float)
    df['GA/GP'] = df['GA/GP'].astype(float)
    df['PP%'] = df['PP%'].astype(float)
    df['PK%'] = df['PK%'].astype(float)
    df['SF/GP'] = df['SF/GP'].astype(float)
    df['SA/GP'] = df['SA/GP'].astype(float)
    df['FOW%'] = df['FOW%'].astype(float)
    df['SA'] = df['SA/GP'] * df['GP'] # calculated approximate SA
    df.drop(['season', 'T', 'OTL', 'points', 'ROW'], axis=1, inplace=True)
    df2['team'] = df2['team'].str.split().str[-1]
    df2.replace({'team': abbr_dict}, inplace = True)
    df2 = df2[['rank', 'team']]
    df2.columns = ['LSrank', 'team']
    return df, df2

def clean_ss(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the season-by-season shot report
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['hits'] = df['hits'].astype(int)
    df['blocked_shots'] = df['blocked_shots'].astype(int)
    df['missed_shots'] = df['missed_shots'].astype(int)
    df['giveaways'] = df['giveaways'].astype(int)
    df['takeaways'] = df['takeaways'].astype(int)
    df['GF'] = df['GF'].astype(int)
    df['SF'] = df['SF'].astype(int)
    df['save%'] = df['save%'].astype(float)
    df['shot%'] = df['GF']/df['SF'] # calculated shot%
    df['PDO'] = df['save%'] + df['shot%'] # calculated PDO
    df.drop(['#', 'season', 'GP', 'W', 'L', 'T', 'OTL', 'points', 'FOW', 'FOL', 'FO', 'FOW%', 'GF'], axis=1, inplace=True)
    return df

def clean_sgf(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the season-by-season goals for report
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['GF55'] = df['GF55'].astype(int)
    df['GF54'] = df['GF54'].astype(int)
    df['GF45'] = df['GF45'].astype(int)
    df = df[['team', 'GF55', 'GF54', 'GF45']]
    return df

def clean_sga(df):
    '''
    pass in appropriate dataframe from WebScraping.py
    will return the clean df for the season-by-season goals against report
    '''
    df['team'] = df['team'].str.split().str[-1]
    df.replace({'team': abbr_dict}, inplace = True)
    df['GA55'] = df['GA55'].astype(int)
    df['GA54'] = df['GA54'].astype(int)
    df['GA45'] = df['GA45'].astype(int)
    df = df[['team', 'GA55', 'GA54', 'GA45']]
    return df

def get_clean_data(df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga):
    '''
    pass in the 8 dataframes, in the order listed above
    will return the cleaned up dataframes
    '''
    df_dts = clean_dts(df_dts)
    df_dp = clean_dp(df_dp)
    df_ds = clean_ds(df_ds)
    df_sts, df_sts_past = clean_sts(df_sts, df_sts_past)
    df_ss = clean_ss(df_ss)
    df_sgf = clean_sgf(df_sgf)
    df_sga = clean_sga(df_sga)
    return df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga

def make_GbG_dfs(df_dts, df_dp, df_ds):
    '''
    pass in the three appropriate dataframes containing game-by-game data
    will return a three dataframes with all relevant information for a GbG basis
    returned dataframes will contain games for past 10, 5, and 2 games, respectively
    '''
    df = pd.merge(df_dts, df_dp, how = 'left', on = ['team', 'opponent', 'date'])
    df = pd.merge(df, df_ds, how = 'left', on ['team', 'opponent', 'date'])
    df['spread'] = df['GF']-df['GA']
    df_last10 = df.loc[df['last10'] == 1]
    df_last5 = df.loc[df['last5'] == 1]
    df_last2 = df.loc[df['last2'] == 1]
    return df_last10, df_last5, df_last2

def make_season_df(df_sts, df_sts_past, df_ss, df_sgf, df_sga):
    '''
    pass in the 5 appropriate dataframes containing season data
    will return a single dataframe containing all season data
    '''
    df = pd.merge(df_sts, df_sts_past, how = 'left', on = ['team'])
    df = pd.merge(df, df_ss, how = 'left', on = ['team'])
    df = pd.merge(df, df_sgf, how = 'left', on = ['team'])
    df = pd.merge(df, df_sga, how = 'left', on = ['team'])
    df['corsi'] = df['SF'] - df['SA']
    return df

if __name__ == '__main__':

    df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga = ws.get_data()

    df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga = \
    get_clean_data(df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga)

    df_last10, df_last5, df_last2 = make_GbG_dfs(df_dts, df_dp, df_ds)
    df_season = make_season_df(df_sts, df_sts_past, df_ss, df_sgf, df_sga)
    
