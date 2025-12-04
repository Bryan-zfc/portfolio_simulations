# portfolio_simulations
I'm practicing portfolio risk simulations.

basic_simulation.py:

This is a very simple Monte Carlo simulation on a portfolio of real historical stocks to estimate portfolio returns, the Value at Risk (VaR), and th Expected Shortfall (ES) over a given time horizon.

First we have to set up our portfolio and parameters (e.g. the number of simulations). Then we download real historical data (using yfinance) and perform the prescribed number of simulations of portfolio returns over the given time horizon. That is, we pick a time horizon number of days randomly from our data, and using the change of the stocks at those dates, we compute what our portfolio returns will look like over this time horizon.

We calculate the loss, the VaR and the ES. The VaR and the ES will be returned, together with a histogram of the simulated portfolio losses. We use sqrt(N) number of bins (where N is the number of simulations), but we might want to change this. 


MC_brownian_simulation.py:
This is similar, but instead of using randomly picked stock prizes from the past to bootstrap, I assume that the stock price S_t behives like a geometric Brownian motion, i.e.: dS_t = μ * S_t * dt + σ * S_t * dW_t. Here, W_t is a Brownian motion, μ is the (annualised) mean of the changes of the stock prices and σ the (annualised) variance. 

If we discretise this, using d(logS) = dS/S - 1/(2*S^2) * dS * dS(and ignoring a lot of the mathematical problems), then we obtain:

S_t+Δt =  S_t * exp( (μ -  σ^2 / 2) * Δt + σ * sqrt(Δt) * N(0,1) ) 

where N(0,1) is a standard Gaussian. 

We hence do a simulation by looking at the most recent stock prices, and simulation the behaviour using the formula above on what the prize will look like after the amount of horizon days. Then, we return the VaR, the ES and a histogram showing the simulated portfolio losses. 
