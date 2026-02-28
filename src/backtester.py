import numpy as np
import matplotlib.pyplot as plt

def backtest(regimes, returns):

    strategy = []
    for r, ret in zip(regimes, returns):
        if r == 0:
            strategy.append(ret)
        else:
            strategy.append(0)

    strategy = np.array(strategy)
    cumulative_strategy = np.cumprod(1 + strategy)
    cumulative_market = np.cumprod(1 + returns)

    plt.plot(cumulative_strategy, label="Strategy")
    plt.plot(cumulative_market, label="Buy & Hold")
    plt.legend()
    plt.show()
