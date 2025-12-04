import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from math import sqrt

#Portfolia data
tickers = ["ABN.AS","ASML.AS","KPN.AS","UNA.AS"] 
weights = np.array([0.1,0.5,0.2,0.2])
initial_value = 20_000

#Start and end dates of data, time horizon and number of simulations
startDate = "2020-01-01"
endDate = None
time_horizon = 30
n_sims = 10000

#Fetch prices and percentual changes between days
data = yf.download(tickers, start=startDate, end=endDate, auto_adjust=False)["Adj Close"].dropna()
returns = data.pct_change().dropna()

#Do n_sims number of simulations and compute how the portfolio would change
sim_portfolio_returns = np.zeros(n_sims)
for i in range(n_sims):
    index = np.random.randint(0, len(returns), size=time_horizon)
    total_returns = (1 + returns.values[index, :]).prod(axis = 0) - 1
    sim_portfolio_returns[i] = np.dot(weights, total_returns)

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

#Plot the simulations in a histogram. The number of bins can be adjusted.
plt.hist(sim_portfolio_returns, bins=int(sqrt(n_sims)))
plt.title(f"Portfolio returns ({time_horizon}-day horizon)")
plt.xlabel("Portfolio return")
plt.ylabel("Frequency")
plt.show()

