# -*- coding: utf-8 -*-

#imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import yfinance as yf


#set up weights and tickers and time frame
bench_tickers = ['GLD']
tickers = ['AAPL', 'TSLA', 'NFLX','AMZN']
weights = [0.25, 0.25, 0.25, 0.25]  
data = yf.download(tickers, start='2024-01-01', end='2025-01-01')
benchdata=yf.download(bench_tickers,start='2024-01-01', end='2025-01-01')

#calculate returns of portfolio and bench
returns = data.pct_change().dropna()
returns2 = returns.iloc[:, :len(tickers)]
bench_returns=benchdata.pct_change().dropna()
bench_returns2=bench_returns.iloc[:,0]
#array of returns for each day across whole portfolio
portfolio_returns = (returns2 * weights).sum(axis=1)
#compounds returns of portfolio and bench over time 
cumulative = (1 + portfolio_returns).cumprod() * 100
bench_cumulative=(1+bench_returns2).cumprod() * 100

#plots cumulative for both on an graph
plt.figure(figsize=(10,6))
plt.plot(cumulative, label='Mock Portfolio', linewidth=2, color='magenta')
plt.plot(bench_cumulative , label='Benchmark (GLD)',color='gold')
plt.title('Mock Portfolio vs Benchmark (GLD)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

#calculates anual volatility and returns to calc sharpe ratio 
#252 as trading days only, sqrt as per formula
annual_return = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = annual_return / annual_volatility
#same thing with only negative returns to calc sortino ratio
downside_returns = portfolio_returns[portfolio_returns < 0]
downside_vol = downside_returns.std()*(252**0.5)
sortino_ratio = annual_return / downside_vol

#prints all these calculated values
print(f"Annualised Return: {annual_return:.2%}")
print(f"Annualised Volatility: {annual_volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Sortino Ratio: {sortino_ratio:.2f}")

#same as cumulative just not x100
cumulative_returns = (1 + portfolio_returns).cumprod()
#cumulative maximum, keeps track of biggest drop
rolling_max = cumulative_returns.cummax()
#calculates peak to trough loses
drawdown = (cumulative_returns - rolling_max) / rolling_max
#picks the biggest one and prints
max_drawdown = drawdown.min()
print(f"Maximum Drawdown: {max_drawdown:.2%}")

#plots drawdowns by date
plt.figure(figsize=(10,6))
plt.plot(drawdown, color='magenta')
plt.title('Portfolio Drawdowns')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.show()

import seaborn as sns
#correcation between returns
corr = returns2.corr()
#plots these correlations
sns.heatmap(corr, annot=True, cmap='rainbow')
plt.title("Asset Correlations")
plt.show()


#calculates different metrics of portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    portfolio_return = np.dot(weights, mean_returns) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe

#calcs portfolio perf and uese [2] to find just sharpe then takes - of it
def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]
#mean returns and covarience matrix of returns, setting up inputs for funcs
mean_returns = returns2.mean()
cov_matrix = returns2.cov()

#number of tickers
num_assets = len(tickers)
#sets weights as equal initially
init_guess = num_assets * [1. / num_assets]
#sets weights to be 0 to 1 for optimiser
bounds = tuple((0, 1) for asset in range(num_assets))
#type=eq: makes sharpe closest to 0, lambda:must sum to 1
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
#finds optimal with bounds, constraints and arguments for negsharp func
optimized = minimize(neg_sharpe, init_guess,
                     args=(mean_returns, cov_matrix, 0.01),
                     method='SLSQP', bounds=bounds, constraints=constraints)
#gives array of optimised weights and prints
opt_weights = optimized.x
print("Optimized Weights:", opt_weights)

#portfolio perf function with optimised weights and prints
opt_return, opt_vol, opt_sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix)
print(f"Optimized Annual Return: {opt_return:.2%}")
print(f"Optimized Annual Volatility: {opt_vol:.2%}")
print(f"Optimized Sharpe Ratio: {opt_sharpe:.2f}")

def simulate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate=0.01):
    results = np.zeros((3, num_portfolios))#array of zeros 3xnum_portfolios
    weight_records = []#creates empty list
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)#creates random set of weights
        weights /= np.sum(weights)#ensures weights sum to 1
        weight_records.append(weights)#adds array onto the list
        
        #port perf func for each set of weights
        ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0,i] = ret #stores 3 results in 1 array [results]
        results[1,i] = vol
        results[2,i] = sharpe
    
    return results, weight_records #spits out results and weights list
#runs that whole function with 5000 portfolios
results, weight_records = simulate_random_portfolios(10000, mean_returns, cov_matrix)
#vol = xaxis, retur= yaxis, sharpe = colour of point
plt.figure(figsize=(10,6))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='nipy_spectral')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficiency')

#plot the optimal portfolio on the graph
plt.scatter(opt_vol, opt_return, c='black', marker='*', s=200, label='Optimal Portfolio')
plt.legend()
plt.show()

#calculates cumulative returns for the optimised weights
optimised_returns = data.pct_change().dropna()
optimised_returns2 = returns.iloc[:, :4]
optimised_portfolio_returns = (optimised_returns2 * opt_weights).sum(axis=1)
optimised_cumulative = (1 + optimised_portfolio_returns).cumprod() * 100

#plots bench, initial and optimised all on the same axis
plt.figure(figsize=(10,6))
plt.plot(optimised_cumulative, label='Optimised Portfolio', linewidth=2, color='cyan')
plt.plot(bench_cumulative , label='Benchmark (GLD)',color='gold')
plt.plot(cumulative, label='Initial Mock Portfolio', linewidth=2, color='magenta')
plt.title('Mock Portfolios vs Benchmark (GLD)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

#create pie chart of weights
plt.figure(figsize=(8,8))
plt.pie(opt_weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=['#FFB3BA', '#BAE1FF', '#BAFFC9', '#FFFFBA'])
plt.title("Optimized Portfolio Weights")
plt.show()
