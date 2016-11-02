import WebScraping as ws
import pandas as pd
import numpy as np

def get_data():

    date1 = '2016-10-12' # in format YYYY-MM-DD
    date2 = '2016-10-31' # in format YYYY-MM-DD
    season1 = '20162017' # in format YYYYyyyy (ie: 20162017)
    season2 = '20162017' # in format YYYYyyyy (ie: 20162017)
    season3 = '20152016' # past season
    season4 = '20152016' # past season

    Date_TeamSummary_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=teamsummary&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=wins,points'.format(date1, date2)
    Date_Penalties_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=penalties&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=penaltyMinutes'.format(date1, date2)
    Date_Shots_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=realtime&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=hits'.format(date1, date2)
    Season_TeamSummary_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=teamsummary&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=points,wins,gamesPlayed'.format(season1, season2)
    Season_TeamSummary_past_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=teamsummary&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=points,wins,gamesPlayed'.format(season3, season4)
    Season_Shots_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=realtime&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=hits'.format(season1, season2)
    Season_GF_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=goalsbystrength&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=goalsFor'.format(season1, season2)
    Season_GA_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=goalsagainstbystrength&reportType=season&seasonFrom={}&seasonTo={}&filter=gamesPlayed,gte,&sort=goalsAgainst'.format(season1, season2)

    soup_dts = ws.get_soup(Date_TeamSummary_url)
    df_dts = ws.Date_TS(soup_dts)

    soup_dp = ws.get_soup(Date_Penalties_url)
    df_dp = ws.Date_Penalties(soup_dp)

    soup_ds = ws.get_soup(Date_Shots_url)
    df_ds = ws.Date_Shots(soup_ds)

    soup_sts = ws.get_soup(Season_TeamSummary_url)
    df_sts = ws.Season_TS(soup_sts)
    soup_sts_past = ws.get_soup(Season_TeamSummary_past_url)
    df_sts_past = ws.Season_TS(soup_sts_past)

    soup_ss = ws.get_soup(Season_Shots_url)
    df_ss = ws.Season_Shots(soup_ss)

    soup_sgf = ws.get_soup(Season_GF_url)
    df_sgf = ws.Season_GF_Strength(soup_sgf)

    soup_sga = ws.get_soup(Season_GA_url)
    df_sga = ws.Season_GA_Strength(soup_sga)

    return df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga

if __name__ == '__main__':

    df_dts, df_dp, df_ds, df_sts, df_sts_past, df_ss, df_sgf, df_sga = get_data()
