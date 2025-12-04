# portfolio_simulations
I'm practicing portfolio risk simulations.

basic_simulation.py:
This is a very simple Monte Carlo simulation on a portfolio of real historical stocks to estimate portfolio returns, the Value at Risk (VaR), and th Expected Shortfall (ES) over a given time horizon.
First we have to set up our portfolio and parameters (e.g. the number of simulations). Then we download real historical data (using yfinance) and perform the prescribed number of simulations of portfolio returns over the given time horizon. 
We calculate the loss, the VaR and the ES. The VaR and the ES will be returned, together with a histogram of the simulated portfolio losses. We use sqrt(N) number of bins (where N is the number of simulations), but we might want to change this. 
