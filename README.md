# A Quick Overview
GOAL: To predict the winner of NHL matchups.

HOW: After scraping data from the NHL website and manipulating it into a workable format, it was then passed through a number of models until an optimal model was found.

RESULTS: The MLP Regressor produced the best results with roughly 60% accuracy.

# The Process: Data Cleaning
First, I had to scrape the game-by-game data from the NHL website for both last season (2015-2016) and the current season thus far (2016-2017). The code had to be written in a way that could be easily updated, as the process would have to be repeated every time updated predictions were desired. BeautifulSoup and Selenium were both required for this process.

After scraping the data, I had three separate data tables for each season: a team summary, a penalty report, and a shot report. These needed to be cleaned up, but then, from these tables, I had to create two intermediate tables before I got a final dataframe that I could work with.

The first table:
A full game-by-game listing with home team, away team, date, an indicator for whether or not the home team won, and then the goal spread in relation to the home team.

The second table:
A game-by-game listing for each team. This table had a team name, a date, and all of the team stats from that game. These stats included things like the goal spread, number of goals, number of shots, penalty minutes, number of hits, etc. It was here that certain engineered features were calculated, like PDO (save % + shot %) and corsi (shots for - shots against).

The final table:
This table was a combination of the above two tables. It was in the format of the the first table, but then average statistics for each team followed, and these average statistics were retrieved from the second table. For example, if we wanted to know the average stats of a team for the five most recent games prior to a matchup, we would find those numbers in this second table. The final table looked something like this:

![Final table example](/images/table.png)

# The Process: Models
With this dataset, I had a number of questions that I had to answer: <br />
1) When calculated the average statistics, was it best to look at 2 games prior to the matchup? 5 games, 10 games, or 15 games? <br />
2) Was it best to look at team stats separately, or as a ratio? <br />
3) Should I predict win/loss? Or predict goal spread and infer win/loss? <br />
4) Which variables should I keep? <br />
5) Should I train the model on games from the current season, last season, or both? <br />
6) Which model produced the best results?

To answer all of these questions, I ran <b>a lot</b> of models.

The following models were tested: <br />
Logistic Regression <br />
Linear Regression <br />
Lasso Regression <br />
Ridge Regression <br />
Elastic Net Regression <br />
Stochastic Gradient Descent Regression <br />
Random Forest Classifier <br />
Random Forest Regressor <br />
XGBoost Classifier <br />
XGBoost Regressor <br />
MLP Classifier <br />
MLP Regressor <br />
Gradient Boosting Classifier <br />
Gradient Boosting Regressor <br />
SVM Classifier <br />
SVM Regressor <br />
Naive Bayes Classifier <br />

Each of these models was tested with average data in the following formats: <br />
2 games, home/away stats separate <br />
5 games, home/away stats separate <br />
10 games, home/away stats separate <br />
15 games, home/away stats separate <br />
5 games, home/away stats as a ratio <br />
10 games, home/away stats as a ratio <br />
15 games, home/away stats as a ratio <br />

I used KFold cross validation with 5 folds and trained the models on data from last season. From here, I chose a number of models to look at more in depth. I pickled these models and then introduced brand new data from this season.

# The Process: Results

While there were clear distinctions between the models last season, all models performed similarly with data from this season. The MLP Regressor (with average data from 5 games prior, home/away stats separate) was slightly better, with an accuracy of roughly 60%. But logistic and linear models were within 3% accuracy. Please see the graphic below for reference. This illustrates the average accuracy of selected models as the season progresses. Please note that this was made on November 15, 2016. If an update graphic is desired, data will have to be refreshed.

![Cumulative Accuracy over time](/images/CumulativeAccuracy.png)

Looking at last season, it seems like there is a clear "optimal model". But please remember that the model was trained on this data, and this is an example of overfitting. This becomes clear when looking at the graph for this season, where it is the worst performing model. This was a common problem with a lot of my models that I could not seem to fix, so these were dismissed.

# Future Steps
Achieving an accuracy score is impressive. It is far better than a 50/50 coin flip, and it is right in line with other similar studies that I have found. I do believe, however, that this score can be improved, as my model is not accounting for factors I think could prove to be important.

1) <b>Roster Changes.  </b>  If a team's best player is injured, the team likely will not perform as well. <br />
2) <b>Number of days since last game.  </b>  A well rested team will likely perform better than a team that played the night before. <br />
3) <b>Breakout of stats for a team at home vs. away.</b>  Maybe a team performs extremely well at home, and not so well away. By only looking at 5 games prior, these stats might not reflect how they actually play. <br />
4) <b>Overtime play.</b>  Maybe things like overtime wins and shootout losses should be considered differently. <br />
5) <b>5-on-5 stats.</b>  Next time, I would only look at stats where both teams are at full strength. Goals counted on powerplay or penalty kill might skew the results. <br />
6) <b>Team rank.</b>  I would factor in a the rank of the team at the time of the matchup. Again, by only looking at stats from 5 games prior, this might not be reflective of a team's true ability.

# Code How To
<b> WebScraping </b> <br />
This file is used to scrape data from the NHL website. Line 156 specifies the start of the current season, and it will scrape all data from this date to the current date. It is currently set up in a way where the number of pages in line 171 needs to be updated prior to running the code. To do this, go to www.nhl.com/stats, and look at game-by-game data for team summary given the dates specified in the code. This may be updated later.

<b> DataCleaning </b> <br />
After updating the number of pages in WebScraping.py, run this file without changing anything. This will update 7 csv files found in the "data" folder. Three of these are the tables scraped from the NHL website: dts (team summary), dp (penalties), and ds (shots). The other four csv files are the average cumulative stats for the past n games: final2, final5, final10, and final15.  

<b> ModelExploration</b> <br />
This file is simply exploring different models while using a KFold cross validation with 5 folds. It is not used at all in the final outcome of the project.

<b> FinalModel_xxxxx</b>
There are a number of these files, each for a different final model that was explored. FinalModel_MLPR.py is the <b>actual</b> final model used to make predictions on future games. At this point, all other files should be up-to-date, so if you would like to look at predictions for future games, they will be added in this file. Please be careful NOT to add past games.

Starting on line 118, copy this line of code for each game that you want to add. Home team is the first thing listed. Please use the three letter abbreviation for the team. If you do not know the abbreviation, please see the dictionary at the beginning of the DataCleaning file. The away team is listed next, followed by the date in YYYY-MM-DD format. It is crucial that home team and away team are listed in that order. After the code runs, you can look at the most recent games by typing in final.tail(). This will also show you the average accuracy of the model at this point in the season. Although, please note that the most accurate accuracy will be reported with the most current <b>completed</b> game.

<b> ModelVisualization</b> <br />
This will produce the updated graphic shown above. 
