
from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta
from selenium import webdriver
from selenium.webdriver.common.by import By

# make sure that you have Google Chrome and selenium installed!

def get_soup(url, page):
    '''
    pass in a url
    will return html output of BeautifulSoup
    uses selenium
    '''
    d = webdriver.Chrome()
    # this will launch a new Chrome browser (maybe multiple)
    # don't exit out until process is finished running
    d.get(url)
    d.find_element(By.XPATH, "//select[@class='pager-select']/option[text()='{}']".format(page)).click()
    result = d.page_source
    soup = BeautifulSoup(result, 'html.parser')
    d.close()
    return soup

def Date_TS(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    this gets us the df for the game-by-game team summary
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['#', 'team', 'game', 'opponent', 'GP', 'W', 'L', 'T', 'OTL', 'points', \
    'GF', 'GA', 'SF', 'SA', 'PPG', 'PP', 'PP%', 'timesSH', 'PPGA', 'PK%', \
    'FOW', 'FOL', 'FOW%']
    mat = np.ndarray((0,23))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df

def Date_Penalties(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    this gets us the df for the game-by-game penalties report
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['#', 'team', 'game', 'opponent', 'GP', 'W', 'L', 'T', 'OTL', 'points', \
    'PIM', 'penalties', 'minor', 'major', 'misconduct', 'game_misconduct', 'match']
    mat = np.ndarray((0,17))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df

def Date_Shots(soup):
    '''
    pass in html output from BeautifulSoup
    will return a pandas dataframe
    this gets us the df for the game-by-game shots report
    '''
    full_table = soup.find("table", {"class": "stat-table"})
    cols = ['#', 'team', 'game', 'opponent', 'GP', 'W', 'L', 'T', 'OTL', 'points', \
    'hits', 'blocked_shots', 'giveaways', 'takeaways', 'FOW', 'FOL', 'FO', \
    'FOW%', 'SF', 'GF', 'save%']
    mat = np.ndarray((0,21))
    lst = soup.find_all('tr', attrs = {'class': 'standard-row'})
    for i in lst:
        row = [elem.text for elem in i.find_all('td')]
        mat = np.vstack((mat, row))
    df = pd.DataFrame(mat,columns = cols)
    return df

def Date_TS_loop(url, num_pages):
    '''
    pass in url and number of pages that are in the table (found below the table at the URL)
    will return a dataframe with all of the pages together
    gives us the df for the game-by-game team summary (total)
    '''
    # creating a dictionary of soup outputs
    # length should be equal to number of pages in table
    soup_dct = {}
    for i in range(1,num_pages+1):
        soup_dct[i] = get_soup(url, page=i)
    # creating a dictionary of dataframes
    # length should be equal to number of pages in table
    df_dct = {}
    for p, soup in soup_dct.items():
        df_dct[p] = Date_TS(soup)
    # appending all dataframes together
    for i in range(2,num_pages+1):
        df_dct[1] = df_dct[1].append(df_dct[i])
    df_dts = df_dct[1]
    return df_dts
    del soup_dct
    del df_dct

def Date_Penalties_loop(url, num_pages):
    '''
    pass in url and number of pages that are in the table (found below the table at the URL)
    will return a dataframe with all of the pages together
    gives us the df for the game-by-game penalties report (total)
    '''
    # creating a dictionary of soup outputs
    # length should be equal to number of pages in table
    soup_dct = {}
    for i in range(1,num_pages+1):
        soup_dct[i] = get_soup(url, page=i)
    # creating a dictionary of dataframes
    # length should be equal to number of pages in table
    df_dct = {}
    for p, soup in soup_dct.items():
        df_dct[p] = Date_Penalties(soup)
    # appending all dataframes together
    for i in range(2,num_pages+1):
        df_dct[1] = df_dct[1].append(df_dct[i])
    df_dp = df_dct[1]
    return df_dp
    del soup_dct
    del df_dct

def Date_Shots_loop(url, num_pages):
    '''
    pass in url and number of pages that are in the table (found below the table at the URL)
    will return a dataframe with all of the pages together
    gives us the df for the game-by-game shots report (total)
    '''
    # creating a dictionary of soup outputs
    # length should be equal to number of pages in table
    soup_dct = {}
    for i in range(1,num_pages+1):
        soup_dct[i] = get_soup(url, page=i)
    # creating a dictionary of dataframes
    # length should be equal to number of pages in table
    df_dct = {}
    for p, soup in soup_dct.items():
        df_dct[p] = Date_Shots(soup)
    # appending all dataframes together
    for i in range(2,num_pages+1):
        df_dct[1] = df_dct[1].append(df_dct[i])
    df_ds = df_dct[1]
    return df_ds
    del soup_dct
    del df_dct

def get_data():

    date1 = '2016-10-12' # start date of current season
    date2 = datetime.date.today().strftime('%Y-%m-%d') # in format YYYY-MM-DD, this is the current date

    Date_TeamSummary_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=teamsummary&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=wins,points'.format(date1, date2)
    Date_Penalties_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=penalties&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=penaltyMinutes'.format(date1, date2)
    Date_Shots_url = 'http://www.nhl.com/stats/team?aggregate=0&gameType=2&report=realtime&reportType=game&startDate={}&endDate={}&filter=gamesPlayed,gte,&sort=hits'.format(date1, date2)

    # soup = get_soup(Date_TeamSummary_url, page = 1)
    # pages = soup.find('select', attrs = {'class': 'pager-select'})
    # num_pages = int(pages.text[-1])

    # df_dts = Date_TS_loop(Date_TeamSummary_url, num_pages)
    # df_dp = Date_Penalties_loop(Date_Penalties_url, num_pages)
    # df_ds = Date_Shots_loop(Date_Shots_url, num_pages)

    df_dts = Date_TS_loop(Date_TeamSummary_url, 9)
    df_dp = Date_Penalties_loop(Date_Penalties_url, 9)
    df_ds = Date_Shots_loop(Date_Shots_url, 9)

    return df_dts, df_dp, df_ds

if __name__ == '__main__':

    df_dts, df_dp, df_ds = get_data()
