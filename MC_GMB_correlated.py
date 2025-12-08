import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import sqrt

#Assets which you want in your portfolio
tickers = ["NVDA","BRK-B","XAIX.DE","IAU"] 
n_tickers = len(tickers)
initial_value = 100_000

#Start and end dates of data, time horizon and number of simulations
startDate = "2020-01-01"
endDate = None
time_horizon = 20
n_sims = 10000


#Fetch prices and percentual changes between days
data = yf.download(tickers, start=startDate, end=endDate, auto_adjust=False)["Adj Close"].dropna()
returns = data.pct_change().dropna()

#Compute the expected annual return, volatility (i.e mean and std) and the correlation matrix.
mu = returns.mean().values * 252        
sigma = returns.std().values * np.sqrt(252) 
corr = returns.corr().values               # correlation matrix
corr_matrix = np.outer(sigma, sigma) * corr # annualised correlation matrix
cov = np.outer(sigma, sigma) * corr
d_t = 1/252

# Compute the optimal weights for your assets using optimzation problems

R_target = 0.10          # target annual return
max_weight = 0.5          # maximum that you want to invest in one single asset

def portfolio_variance(w):
    return w @ cov @ w 

# Constraints for the optimization: sum of weights equals 1 (i.e. invest everything), expected return >= target return and the weights are 0 <= w <= max_weigth
constraints = (
    {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    {"type": "ineq", "fun": lambda w: w @ mu - R_target},
)
# Bounds (cannot short sell, i.e. weights >= 0)
bounds = [(0, max_weight)] * n_tickers

# Starting guess needed for the minimize function below
w_0 = np.ones(n_tickers) / n_tickers

# SLSQP = Sequential Least Squares Programming
res = minimize(portfolio_variance, w_0, method="SLSQP",
               constraints=constraints, bounds=bounds)

optimal_weights = res.x 

print("Optimal weights:", optimal_weights) #prints the opimal weights for the selected assets


#Do n_sims number of simulations on how your portfolio would change using Brownian Motion and using the Cholesky decomposition of the correlation matrix to get correlated Geometric Brownian Motion
L = np.linalg.cholesky(corr_matrix)
sim_portfolio_returns = np.zeros(n_sims)

for i in range(n_sims):
    S_t = data.iloc[-1].values #Most recent prices
    for t in range(time_horizon):
        Z = np.random.randn(n_tickers)
        correlated_Z = L @ Z
        S_t = S_t * np.exp( ((mu - 0.5*sigma ** 2)*(d_t) + sigma * np.sqrt(d_t) * correlated_Z )    )
        path_returns = (S_t / data.iloc[-1].values) - 1 
        sim_portfolio_returns[i] = np.dot(optimal_weights, path_returns)


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

