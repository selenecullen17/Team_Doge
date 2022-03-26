def hello_world():
    print("hello world!")
 # -*- coding: utf-8 -*-
hello_world()
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from tiingo import TiingoClient                       # Stock prices.
import quandl                                         # Economic data, futures prices, ...

# API keys:
tiingo = TiingoClient({'api_key':'349dd4117e5d1ed71a22f0207a8e439cf3f7b06b'})
quandl.ApiConfig.api_key = '_ci7hzC_sPkspKWz2yzt'                      # Replace the XXXX with your API key (must be inside the ' ').

import matplotlib.pyplot as plt                        # Basic plot library.
plt.style.use('ggplot')                                # Make plots look nice.


def prices(tickers,start_date,end_date):
    prices  = tiingo.get_dataframe(tickers, start_date, end_date, metric_name='adjClose')
    prices.index = pd.to_datetime(prices.index).tz_convert(None)
    return prices

def returns(tickers,start_date,end_date):
    prices  = tiingo.get_dataframe(tickers, start_date, end_date, metric_name='adjClose')
    prices.index = pd.to_datetime(prices.index).tz_convert(None) 
    
    #find the returns of the tickers in prices
    r = prices.pct_change()
    
    return r

def rp(tickers,start_date,end_date):
    # first pull the data you need from tiingo
    prices  = tiingo.get_dataframe(tickers, start_date, end_date, metric_name='adjClose')
    prices.index = pd.to_datetime(prices.index).tz_convert(None) 

    
    #find the returns of the tickers in prices
    r = prices.pct_change()
    
    #assuming the risk free rate for now
    #gathering important statistics for analysis
    rf = 0.011
    risk_premiums = r.mean() * 252 - rf       
    
    return risk_premiums

def vol(tickers,start_date,end_date):
        # first pull the data you need from tiingo
    prices  = tiingo.get_dataframe(tickers, start_date, end_date, metric_name='adjClose')
    prices.index = pd.to_datetime(prices.index).tz_convert(None) 
   
    
    #find the returns of the tickers in prices
    r = prices.pct_change()
    #find the volatility      
    vol           = r.std()  * 252**0.5
    
    return vol

def cov(tickers,start_date,end_date):
        # first pull the data you need from tiingo
    prices  = tiingo.get_dataframe(tickers, start_date, end_date, metric_name='adjClose')
    prices.index = pd.to_datetime(prices.index).tz_convert(None) 
    
    
    #find the returns of the tickers in prices
    r = prices.pct_change()
    #find the covariance      
    cov           = r.cov()  * 252
    
    return cov

def corr(tickers,start_date,end_date):
        # first pull the data you need from tiingo
    prices  = tiingo.get_dataframe(tickers, start_date, end_date, metric_name='adjClose')
    prices.index = pd.to_datetime(prices.index).tz_convert(None) 

    
    #find the returns of the tickers in prices
    r = prices.pct_change()
    #find the covariance      
    corr          = r.corr()
    
    return corr

def mean_vol_plot(tickers,risk_premiums,vol,xmax,ymin,ymax):
    
    assets = pd.DataFrame()                        
    assets['Risk_premium'] = risk_premiums
    assets['Volatility']   = vol    
    assets['Color']        = 'orange'
    assets['Size']         = 150

    all_portfolios = assets
    all_portfolios
    
    graph = all_portfolios.plot.scatter('Volatility','Risk_premium', color=all_portfolios.Color, s=all_portfolios.Size, xlim=(0,xmax), ylim=(ymin,ymax))

    # add ticker symbols to plot:
    for s in tickers:                                 # loop over sectors
        x = all_portfolios.loc[s,'Volatility']+0.005  # get x-coordinate from table (+ offset so the labels don't overlap with points)
        y = all_portfolios.loc[s,'Risk_premium']      # get y-coordinate from table
        graph.text(x,y,s)                             # add the ticker symbol to the graph
    return graph

def efficient_frontier(tickers,risk_premiums,vol,cov,xmax,ymin,ymax):
    assets = pd.DataFrame()                        
    assets['Risk_premium'] = risk_premiums
    assets['Volatility']   = vol    
    assets['Color']        = 'orange'
    assets['Size']         = 150

    all_portfolios = assets
    all_portfolios
    
   # here we are building the simulation that finds the possible portfolios
    n_portfolios = 1000

    random_arrays  = [ np.random.uniform(0,1,len(tickers)) for i in range(0,n_portfolios) ]
    random_weights = [ ra/ra.sum() for ra in random_arrays ]
    
    random = pd.DataFrame()
    random['Risk_premium'] = [ w.dot(risk_premiums)   for w in random_weights ]
    random['Volatility']   = [ w.dot(cov).dot(w)**0.5 for w in random_weights ]
    random['Color']        = 'gray'
    random['Size']         = 10
    random['sharpe']       = random['Risk_premium']/random['Volatility']

    all_portfolios = pd.concat([random, assets])
    
    graph = all_portfolios.plot.scatter('Volatility','Risk_premium', color=all_portfolios.Color, s=all_portfolios.Size, xlim=(0,xmax), ylim=(ymin,ymax))

    # add ticker to plot:
    for s in tickers:                                 # loop over sectors
        x = all_portfolios.loc[s,'Volatility']+0.005  # get x-coordinate from table (+ offset so the labels don't overlap with points)
        y = all_portfolios.loc[s,'Risk_premium']      # get y-coordinate from table
        graph.text(x,y,s)                             # add the ticker symbol to the graph
    return graph

def max_sharpe_weights(tickers,risk_premiums,vol,cov):
    #we have to do the simulation again, I'm sure this could also be done with a nested function but that's ok
    assets = pd.DataFrame()                        
    assets['Risk_premium'] = risk_premiums
    assets['Volatility']   = vol    
    assets['Color']        = 'orange'
    assets['Size']         = 150

    all_portfolios = assets
    all_portfolios
    
    n_portfolios = 1000

    random_arrays  = [ np.random.uniform(0,1,len(tickers)) for i in range(0,n_portfolios) ]
    random_weights = [ ra/ra.sum() for ra in random_arrays ]
    
    random = pd.DataFrame()
    random['Risk_premium'] = [ w.dot(risk_premiums)   for w in random_weights ]
    random['Volatility']   = [ w.dot(cov).dot(w)**0.5 for w in random_weights ]
    random['Color']        = 'gray'
    random['Size']         = 10
    random['sharpe']       = random['Risk_premium']/random['Volatility']

    all_portfolios = pd.concat([random, assets])
    
    #now we have to do some crazy stuff, basically I am creating a data frame where all of the entries 
    #are themselves lists of possible weights
    #I am going to parse through the dataframe to find the weights that correspond to the maximum sharpe ratio
    random['weights'] = random_weights
    wgts = random[(random['sharpe'] == random.sharpe.max())].weights.astype(str)

    rp = random[(random['sharpe'] == random.sharpe.max())].Risk_premium
    #now I have found the weights, but they are in string form. We want them to be a panda series
    #here i am turning the string weights into a list
    lw = list(wgts)
    #finding the first entry in the list(the first entry is always a quote)
    ws = str(lw[0])
    # replacing the quote with a blank
    ws = ws.replace('[','')
    ws = ws.replace(']','')
    #deliniating the list values by spaces
    lw = ws.split(' ')
    #filtering out the bad data
    lw = list(filter(None, lw))
    #Finally turning the list into a numpy array
    nw = np.array(lw)
    nw = nw.astype(float)
    
    #now we have to turn the numpy array into a panda series 
    
    sharpe_weights                  = pd.DataFrame()
    sharpe_weights['Risk_Premiums'] = risk_premiums
    sharpe_weights['Weights']       = nw
    sharpe_weights['weighted_ret']  = sharpe_weights['Weights']*sharpe_weights['Risk_Premiums']
    
    return sharpe_weights


#don't forget to turn this into a variable when you run the function!

def sharpe_ratio_value(tickers,risk_premiums,vol,cov):
    #we have to do the simulation again, I'm sure this could also be done with a nested function but that's ok
    assets = pd.DataFrame()                        
    assets['Risk_premium'] = risk_premiums
    assets['Volatility']   = vol    
    assets['Color']        = 'orange'
    assets['Size']         = 150

    all_portfolios = assets
    all_portfolios
    
    n_portfolios = 1000

    random_arrays  = [ np.random.uniform(0,1,len(tickers)) for i in range(0,n_portfolios) ]
    random_weights = [ ra/ra.sum() for ra in random_arrays ]
    
    random = pd.DataFrame()
    random['Risk_premium'] = [ w.dot(risk_premiums)   for w in random_weights ]
    random['Volatility']   = [ w.dot(cov).dot(w)**0.5 for w in random_weights ]
    random['Color']        = 'gray'
    random['Size']         = 10
    random['sharpe']       = random['Risk_premium']/random['Volatility']

    all_portfolios = pd.concat([random, assets])
    
    sharpe_ratio = random.sharpe.max()
    return sharpe_ratio

def max_sharpe_risk_premium(tickers,risk_premiums,vol,cov):
    assets = pd.DataFrame()                        
    assets['Risk_premium'] = risk_premiums
    assets['Volatility']   = vol    
    assets['Color']        = 'orange'
    assets['Size']         = 150

    all_portfolios = assets
    all_portfolios
    
    n_portfolios = 1000

    random_arrays  = [ np.random.uniform(0,1,len(tickers)) for i in range(0,n_portfolios) ]
    random_weights = [ ra/ra.sum() for ra in random_arrays ]
    
    random = pd.DataFrame()
    random['Risk_premium'] = [ w.dot(risk_premiums)   for w in random_weights ]
    random['Volatility']   = [ w.dot(cov).dot(w)**0.5 for w in random_weights ]
    random['Color']        = 'gray'
    random['Size']         = 10
    random['sharpe']       = random['Risk_premium']/random['Volatility']

    all_portfolios = pd.concat([random, assets])
    max_sharpe_risk_premium = random[(random['sharpe'] == random.sharpe.max())].Risk_premium
    return max_sharpe_risk_premium

def max_sharpe_volatility(tickers,risk_premiums,vol,cov):
    assets = pd.DataFrame()                        
    assets['Risk_premium'] = risk_premiums
    assets['Volatility']   = vol    
    assets['Color']        = 'orange'
    assets['Size']         = 150

    all_portfolios = assets
    all_portfolios
    
    n_portfolios = 1000

    random_arrays  = [ np.random.uniform(0,1,len(tickers)) for i in range(0,n_portfolios) ]
    random_weights = [ ra/ra.sum() for ra in random_arrays ]
    
    random = pd.DataFrame()
    random['Risk_premium'] = [ w.dot(risk_premiums)   for w in random_weights ]
    random['Volatility']   = [ w.dot(cov).dot(w)**0.5 for w in random_weights ]
    random['Color']        = 'gray'
    random['Size']         = 10
    random['sharpe']       = random['Risk_premium']/random['Volatility']

    all_portfolios = pd.concat([random, assets])
    max_sharpe_volatility = random[(random['sharpe'] == random.sharpe.max())].Volatility
    return max_sharpe_volatility
    
def var_weight_plot(base_tickers,var_ticker,base_weights,start_date,end_date,n):
    base_weights = np.array(base_weights)
    
    
    base_prices = tiingo.get_dataframe(base_tickers, start_date, end_date, metric_name='adjClose')

    
    base_returns          = base_prices.pct_change()
    weighted_base_returns = base_weights*base_returns
    base_total_ret        = weighted_base_returns.sum(axis = 1)
    
    var_prices   = tiingo.get_dataframe(var_ticker, start_date, end_date, metric_name='adjClose')
    var_prices   = pd.DataFrame(var_prices)
    var_returns  = var_prices.pct_change()
    
    #now create the array of variable weights
    var_weight = []
    for i in range(n):
        var_weight.append(i/n)
    var_weight  = np.array(var_weight)
    
    test_returns                   = pd.DataFrame()
    test_returns['Base_Portfolio'] = base_total_ret
    test_returns['Variable_Sec']   = var_returns
    
    
    Volatility   = []
    Risk_Premium = []

    test_port_returns = []
    #this for loop creates n portfolios with the desired weights and puts the summary statistics into a DF
    for w in var_weight:
        var_weighted_return  = test_returns.Variable_Sec*w
        base_weighted_return = test_returns.Base_Portfolio*(1-w)
    
        total_ret = var_weighted_return + base_weighted_return
        total_ret = pd.DataFrame(total_ret)
    
        return_average    = total_ret.mean()*252
        return_volatility = total_ret.std()*252**0.5
        return_rp         = return_average - 0.011
    
        test_port_returns.append(return_average)
        Volatility.append(return_volatility)
        Risk_Premium.append(return_rp)
    #this makes the numbers inside the arrays floats so that we can plot them
    Risk_Premium = pd.DataFrame(Risk_Premium)
    Volatility   = pd.DataFrame(Volatility)
        
    var_test_results                 = pd.DataFrame()
    var_test_results['var_weight']   = var_weight
    var_test_results['Risk_Premium'] = Risk_Premium
    var_test_results['Volatility']   = Volatility
    var_test_results['Sharpe_Ratio'] = var_test_results.Risk_Premium/var_test_results.Volatility
    var_test_results
  
    #now create the plot 
    graph = var_test_results[['Risk_Premium','Volatility','Sharpe_Ratio']].plot(logy = False)
    
    
    
    return var_test_results,graph


