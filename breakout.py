# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:21:08 2020

@author: Raghav
"""

import pandas as pd
import datetime as dt
import pandas_datareader as pdr
import copy
import numpy as np
import pickle
### Pulling NIFTY-50 2014 data ##
#
##url = "https://www.dalalstreetwinners.com/nifty-50-stocks-2014/8473/"
##dfs = pd.read_html(url)
#
##nifty_old = dfs[0]
##nifty_old.columns = ["Company", "Industry", "Symbol", "ISIEN"]
##nifty_old.drop(0, inplace = True)
#
##string = '.NS'
##tickers = list(map(lambda orig_string: orig_string + string, nifty_old["Symbol"]))
#
##with open('ticker_list.txt', 'wb') as input:
#    #pickle.dump(tickers, input)
#
### Pulling monthly data for tickers from yahoo finance ##
#with open('ticker_list.txt', 'rb') as output:
#    tickers = pickle.load(output)
#
#ohlc_dict = {}
#attempt = 0
#drop = []
#start_date =  dt.date.today()-dt.timedelta(1460)
#end_date = dt.date.today()
#
#while len(tickers) != 0 and attempt <= 5:
#    tickers = [j for j in tickers if j not in drop]
#    for i in range(len(tickers)):
#        try:
#            print("Getting data for", tickers[i])
#            ohlc_dict[tickers[i]] = pdr.get_data_yahoo(tickers[i], start_date, end_date, interval='d' )
#            ohlc_dict[tickers[i]].dropna(inplace = True)
#            drop.append(tickers[i])
#           
#        except:
#           print(tickers[i], ":Failed to get data, retrying..")
#            
#        attempt += 1   
#        
#pickle.dump(ohlc_dict, open("daily_data.pickle", "wb"))
#        
#tickers = ohlc_dict.keys()
#####################Creating environment and data###################

with open('daily_data.pickle', 'rb') as fp:
    ohlc_dict = pickle.load(fp)
    
    
tickers = ohlc_dict.keys()
       
ohlc = copy.deepcopy(ohlc_dict)
state = {}
returns = {}

for ticker in tickers:
    print("Adding required data for", ticker)
    ohlc[ticker]["200 MA"] = ohlc[ticker]["Adj Close"].rolling(200).mean()
    ohlc[ticker]["100 MA"] = ohlc[ticker]["Adj Close"].rolling(100).mean()
    ohlc_dict[ticker]["roll_max_cp"] = ohlc_dict[ticker]["High"].rolling(20).max()
    ohlc_dict[ticker]["roll_min_cp"] = ohlc_dict[ticker]["Low"].rolling(20).min()
    ohlc_dict[ticker]["roll_max_vol"] = ohlc_dict[ticker]["Volume"].rolling(20).max()
    ohlc[ticker].dropna(inplace=True)
    state[ticker] = ""
    returns[ticker] = []
    
    
#########backtesting logic and returns############
stoploss = 0.05

for ticker in tickers:
    for i in range(len(ohlc[ticker])):
        if state[ticker] == "":
            returns[ticker].append(0)
            if ohlc[ticker]["Adj Close"][i] >= ohlc[ticker]["200 MA"][i] and ohlc[ticker]["25 MA"][i] >= ohlc[ticker]["100 MA"][i]:
                state[ticker] = "Buy"
            elif ohlc[ticker]["Adj Close"][i] <= ohlc[ticker]["200 MA"][i] and ohlc[ticker]["100 MA"][i] >= ohlc[ticker]["25 MA"][i]:
                state[ticker] = "Sell"
                
        elif state[ticker] == "Buy":
            if ohlc[ticker]["Adj Close"][i-1] <= (1-stoploss)*ohlc[ticker]["Adj Close"][i]:
                state[ticker] = ""
                returns[ticker].append((ohlc[ticker]["Adj Close"][i]-ohlc[ticker]["Adj Close"][i-1])/ohlc[ticker]["Adj Close"][i])
           
            elif ohlc[ticker]["Adj Close"][i] <= ohlc[ticker]["200 MA"][i] and ohlc[ticker]["100 MA"][i] >= ohlc[ticker]["25 MA"][i]:
                state[ticker] = "Sell"
                returns[ticker].append((ohlc[ticker]["Adj Close"][i]-ohlc[ticker]["Adj Close"][i-1])/ohlc[ticker]["Adj Close"][i])
                
            else:
                returns[ticker].append((ohlc[ticker]["Adj Close"][i]-ohlc[ticker]["Adj Close"][i-1])/ohlc[ticker]["Adj Close"][i])
                
        elif state[ticker] == "Sell":
            if ohlc[ticker]["Adj Close"][i-1] >= (1+stoploss)*ohlc[ticker]["Adj Close"][i]:
                state[ticker] = ""
                returns[ticker].append((ohlc[ticker]["Adj Close"][i-1]-ohlc[ticker]["Adj Close"][i])/ohlc[ticker]["Adj Close"][i-1])
                
            elif ohlc[ticker]["Adj Close"][i] >= ohlc[ticker]["200 MA"][i] and ohlc[ticker]["25 MA"][i] >= ohlc[ticker]["100 MA"][i]:
                state[ticker] = "Buy"
                returns[ticker].append((ohlc[ticker]["Adj Close"][i-1]-ohlc[ticker]["Adj Close"][i])/ohlc[ticker]["Adj Close"][i-1])
                
            else:
                returns[ticker].append((ohlc[ticker]["Adj Close"][i-1]-ohlc[ticker]["Adj Close"][i])/ohlc[ticker]["Adj Close"][i-1])
           
        

    ohlc[ticker]["return"] = np.array(returns[ticker])
        
        
####DEFINING KPI #############
def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    n = len(df)/(252)
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(252)
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd    

# calculating overall strategy's KPIs
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlc[ticker]["return"]
strategy_df["ret"] = strategy_df.mean(axis=1)
print(CAGR(strategy_df))
print(sharpe(strategy_df,0.075))
print(max_dd(strategy_df))

# vizualization of strategy return
(1+strategy_df["ret"]).cumprod().plot()




    



            
            
    
        
    




























    


