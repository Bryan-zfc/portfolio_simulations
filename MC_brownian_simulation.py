import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from math import sqrt

#Portfolia data
tickers = ["ABN.AS","ASML.AS","KPN.AS","UNA.AS"] 
n_tickers = len(tickers)
weights = np.array([0.1,0.5,0.2,0.2]) #Should sum to 1
initial_value = 20_000

#Start and end dates of data, time horizon and number of simulations
startDate = "2020-01-01"
endDate = None
time_horizon = 25
n_sims = 1000


#Fetch prices and percentual changes between days
data = yf.download(tickers, start=startDate, end=endDate, auto_adjust=False)["Adj Close"].dropna()
returns = data.pct_change().dropna()

#Compute the expected annual return and volatility (i.e mean and std) 
mu = returns.mean().values * 252        
sigma = returns.std().values * np.sqrt(252) 
d_t = 1/252

#Do n_sims number of simulations on how your portfolio would change using Brownian Motion
sim_portfolio_returns = np.zeros(n_sims)
for i in range(n_sims):
    S_t = data.iloc[-1].values #Most recent prices
    for t in range(time_horizon):
        Z = np.random.randn(n_tickers)
        S_t = S_t * np.exp( ((mu - 0.5*sigma ** 2)*(d_t) + sigma * np.sqrt(d_t) * Z )    )
        path_returns = (S_t / data.iloc[-1].values) - 1 
        sim_portfolio_returns[i] = np.dot(weights, path_returns)


end_vals = initial_value * (1 + sim_portfolio_returns)
losses = initial_value - end_vals

#Compute the Value at Risk and the Expected Shortfall
def var_es(losses, alpha=0.95):
    VaR = np.quantile(losses, alpha)
    tail = losses[losses >= VaR]
    es = tail.mean() if len(tail)>0 else float("nan")
    return float(VaR), float(es)

#Print the VaR and ES for 95% and 99% respectively
print("VaR95, ES95:", var_es(losses, 0.95))
print("VaR99, ES99:", var_es(losses, 0.99))

plt.hist(sim_portfolio_returns, bins=int(sqrt(n_sims)))
plt.title(f"MC using Brownian motion portfolio returns ({time_horizon}-day horizon)")
plt.xlabel("Portfolio return")
plt.ylabel("Frequency")
plt.show()

