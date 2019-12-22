import markets
import traders
import genetic_traders as gt
import numpy as np
import matplotlib.pyplot as plt

balance = 1000
num_stocks = 200
num_positions = 20

# transaction cost = 0
buy_filters = [gt.MinPriceFilter(1.31)]
buy_ranking = gt.ROCMinMeanRanking(5)
sell_filters = [gt.MinPriceFilter(1.27)]
emergency_sell = 0.62

# transaction cost = 8
#buy_filters = [gt.MaxPriceFilter(1.58)]
#buy_ranking = gt.ROCMinMeanRanking(119)
#sell_filters = [gt.ROCMinMeanFilter(161, 3.05)]
#emergency_sell = 0.104

trader = traders.CustomTrader(balance, num_stocks, num_positions,
                              buy_filters, buy_ranking,
                              sell_filters, emergency_sell)

length = 3000
runs = 100
run_values = np.zeros((runs, length+1))
run_trades = np.zeros((runs, length+1))
for j in range(runs):
    market = markets.NoisyMarket(200, num_stocks, 0, 0.1)
    trader = traders.CustomTrader(balance, num_stocks, num_positions,
                                  buy_filters, buy_ranking,
                                  sell_filters, emergency_sell)
    values = [balance]
    trades = [0]
    for i in range(length):
        trader.update_portfolio(market)
        market.step()
        value = trader.get_value(market)
        values.append(value)
        trades.append(trader.trades)
    run_values[j] = np.array(values)
    run_trades[j] = np.array(trades)

sorted_values = np.sort(run_values, 0)
sorted_trades = np.sort(run_trades, 0)

days = np.arange(length + 1)
fig, ax = plt.subplots(figsize=(12,6), nrows=1, ncols=2)
ax[0].plot(days, np.median(run_values, 0), color='dodgerblue')
ax[0].fill_between(days, sorted_values[9, :],
                   sorted_values[89, :],
                   alpha=0.25, color='dodgerblue')
ax[0].fill_between(days, sorted_values[24, :],
                   sorted_values[74, :],
                   alpha=0.5, color='dodgerblue')
ax[0].set_xlabel('Days')
ax[0].set_ylabel('Trader Value')
#ax[0].set_yscale('log')
ax[0].set_title('Value Over Time')
ax[1].plot(days, np.median(run_trades, 0), color='dodgerblue')
ax[1].fill_between(days, sorted_trades[9, :],
                   sorted_trades[89, :],
                   alpha=0.25, color='dodgerblue')
ax[1].fill_between(days, sorted_trades[24, :],
                   sorted_trades[74, :],
                   alpha=0.5, color='dodgerblue')
ax[1].set_xlabel('Days')
ax[1].set_ylabel('Total Trades')
ax[1].set_title('Trades Over Time')
plt.show()