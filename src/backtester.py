import numpy as np
import matplotlib.pyplot as plt

def backtest(predicted_regimes, returns):

    strategy_returns = []

    for regime, ret in zip(predicted_regimes, returns):
        if regime == 0:
            strategy_returns.append(ret)
        else:
            strategy_returns.append(0)

    strategy_returns = np.array(strategy_returns)

    cumulative_strategy = np.cumprod(1 + strategy_returns)
    cumulative_market = np.cumprod(1 + returns)

    plt.figure(figsize=(10,6))
    plt.plot(cumulative_strategy, label="Regime Strategy")
    plt.plot(cumulative_market, label="Buy & Hold")
    plt.legend()
    plt.title("Strategy vs Market")
    plt.show()
