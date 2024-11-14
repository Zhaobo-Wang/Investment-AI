import numpy as np 
import pandas as pd 
from pandas import DataFrame, Series 
import datetime as dt 
import matplotlib.pyplot as plt 
 
#load data 
datafilename = "./Train_IntraDayData_5minute.xlsx" 
stock_data = pd.read_excel(datafilename, header=[0,1], index_col=0, 
parse_dates=True) 
stock_data.head() 
 
#pre-process 
i=4 #close 
stock_data_1=stock_data.iloc[:,[i,i+5*2,i+5*4,i+5*7,i+5*11,i+5*12,i+5*13,i+5*15,i+5*18,i+5*19]]
stock_data_1 
 
# returns for individual stock 
d_num=82 #49-5min 24-10min 8-30min 
num_assets = len(stock_data_1.columns) 
cov_matrix = stock_data_1.pct_change().apply(lambda x: np.log(1+x)).cov() 
corr_matrix = stock_data_1.pct_change().apply(lambda x: np.log(1+x)).corr() 
# Randomly weighted portfolio's variance 
w=np.ones(num_assets)*(1/num_assets) 
# mean returns for individual companies 
return_mean = stock_data_1.resample('M').last().pct_change().mean() 
# Portfolio returns 
port_er = (w*return_mean).sum() 
# Volatility is given by the standard deviation. 
volat = stock_data_1.pct_change().apply(lambda x: 
np.log(1+x)).std().apply(lambda x: x*np.sqrt(d_num)) 
assets = pd.concat([return_mean, volat], axis=1) 
# Creating a table for visualising returns and volatility of assets 
assets.columns = ['Returns', 'Volatility'] 
assets 
 
p_ret = [] # Define an empty array for portfolio returns 
p_vol = [] # Define an empty array for portfolio volatility 
p_weights = [] # Define an empty array for asset weights 
#np.random.seed(6) 
num_assets = len(stock_data_1.columns) 
num_portfolios = 100000 
 
for portfolio in range(num_portfolios): 
 weights = np.random.random(num_assets) 
 weights = weights/np.sum(weights) 
 p_weights.append(weights) 
 returns = np.sum(return_mean*weights) 
 p_ret.append(returns) 
 volat = np.sqrt(np.dot(weights.T,np.dot(cov_matrix,weights)))*np.sqrt(d_num) 
 p_vol.append(volat) 
 
data = {'Returns':p_ret, 'Volatility':p_vol, 'Weight':p_weights} 
for counter, symbol in enumerate(stock_data_1.columns.tolist()): 
 #print(counter, symbol) 
 data[symbol] = [w[counter] for w in p_weights] 
 
portfolios = pd.DataFrame(data) 
portfolios.head() # Dataframe of the 10000 portfolios created 
 
min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()] 
# idxmin() gives us the minimum value in the column specified.
min_vol_port 
 
# Finding the optimal portfolio 
rf = 0.01 # risk factor 
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-
rf)/portfolios['Volatility']).idxmax()] 
optimal_risky_port 
 
# plotting the minimum volatility portfolio 
plt.subplots(figsize=[8,8]) 
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', 
s=50, alpha=0.7) 
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', 
s=500) 
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', 
marker='*', s=500) 