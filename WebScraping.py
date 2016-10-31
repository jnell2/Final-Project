from pymongo import MongoClient
from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import numpy as np
from selenium import webdriver

# make sure that you have Google Chrome and selenium installed!

def get_soup(url):
    '''
    pass in a url
    will return html output of BeautifulSoup
    '''
    d = webdriver.Chrome()
    # this will launch a new Chrome browser
    # don't exit out until process is finished running
    d.get(url)
    result = d.page_source
    soup = BeautifulSoup(result, 'html.parser')
    return soup

def Date_TS(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['#','team', 'game', 'opponent', 'GP', 'W', 'L', 'T', 'OT', 'points', \
    'GF', 'GA', 'SF', 'SA', 'PPG', 'PP', "PP%", 'timesSH', 'PPGA', "PK%", \
    'FOW', 'FOL', "FOW%"]
    mat = np.ndarray((0,23))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df


if __name__ == '__main__':
    # client = MongoClient()
    # db = client['']
    # table = db['']

    Date_TeamSummary_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=teamsummary&reportType=game&startDate=2016-10-01&endDate=2016-10-31&filter=gamesPlayed,gte,&sort=wins,points'
    Date_Penalties_url = ''
    Date_Shots_url = ''
    Season_TeamSummary_url = ''
    Season_Shots_url = ''

    soup = get_soup(Date_TeamSummary_url)
    df = Date_TS(soup)

    # urls = [Date_TeamSummary_url, Date_Penalties_url, Date_Shots_url, Season_TeamSummary_url, Season_Shots_url]
    # for report in urls:
    #     soup = get_soup(report)
#
# # Date range: 10/01/2016 to 10/31/2016
# # Report: Team Summary
# # Location: Home AND Road (can deduce Home team)

# Date range: 10/01/2016 to 10/31/2016
# Report: Penalties
# Location: Home AND Road (can deduce Home team)

# Date range: 10/01/2016 to 10/31/2016
# Report: Hits, BkS, MsS, Gvwys, Tkwys
# Location: Home AND Road (can deduce Home team)

# Season by Season: 2016-2017
# Report: Team Summary

# Season by Season: 2016-2017
# Report: Hits, BkS, MsS, Gvwys, Tkwys

# Season by Season: 2016-2017
# Report: Leading/Trailing ?????

# Season by Season: 2016-2017
# Report: Out-Shoot/Out-Shot By ?????

# Season by Season: 2016-2017
# Report: Team Goal-Games ?????

# Season by Season: 2016-2017
# Report: Goals by Strength ?????

# Season by Season: 2016-2017
# Report: Goals Against by Strength ?????

# Season by Season: 2016-2017
# Report: Goals by Period ?????
