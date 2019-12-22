# EvoTrading

This repository contains trading agents that use sets of rules to decide when to buy and sell stocks. The rules can be set
manually. Alternatively, a population of rule-based traders can be evolved to maximise returns. Historical NASDAQ stock prices
were used to test and evolve sets of trading rules.

Sets of rules for a trading agent were evolved for two scenarios. In the first, there was no cost for buying or selling a stock.
The returns and number of trades for this trader assuming no transaction costs are shown below.

<p align="center">
  <img src="images/transaction_cost_0.png">
</p>

In the second, there was an £8 cost for buying or selling any amount of a stock. The returns and number of trades for this
trader assuming £8 transaction costs are shown below.

<p align="center">
  <img src="images/transaction_cost_8.png">
</p>
