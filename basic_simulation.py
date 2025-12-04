import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from math import sqrt

#Portfolia data, start and end dates of data, number of simulations, horizon_days, and initial investment money
tickers = ["ABN.AS","ASML.AS","KPN.AS","UNA.AS"] 
weights = np.array([0.3,0.3,0.25,0.15])
startDate = "2020-01-01"
endDate = None
time_horizon = 30
n_sims = 10000
initial_value = 1_000_000

#Fetch prices and percentual changes between days
data = yf.download(tickers, start=startDate, end=endDate, auto_adjust=False)["Adj Close"].dropna()
returns = data.pct_change().dropna()

days_history = len(returns)
sim_portfolio_returns = np.zeros(n_sims)

#Do n_sims number of simulations on how your portfolio would change
for i in range(n_sims):
    index = np.random.randint(0, days_history, size=time_horizon)
    path_returns = (1 + returns.values[index, :]).prod(axis = 0) - 1
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
plt.title(f"Bootstrap portfolio returns ({time_horizon}-day horizon)")
plt.xlabel("Portfolio return")
plt.ylabel("Frequency")
plt.show()

