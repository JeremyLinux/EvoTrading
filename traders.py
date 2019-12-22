import numpy as np

class BuyAndHoldTrader(object):
    
    def __init__(self, balance, num_stocks):
        self.balance = balance
        self.num_stocks = num_stocks
        self.portfolio = np.zeros(num_stocks)
    
    def buy_stock(self, stock_number, payment, market):
        price = market.get_current_prices(1).flatten()[stock_number]
        charge = market.transaction_cost
        if self.balance < charge:
            payment = 0
        elif self.balance < (payment - charge):
            payment = self.balance
        self.balance -= payment
        self.portfolio[stock_number] += ((payment - charge) / price)
    
    def update_portfolio(self, market):
        if self.balance > 0:
            for i in range(self.num_stocks):
                self.buy_stock(i, self.balance / (self.num_stocks - i),
                               market)
    
    def get_value(self, market):
        current_prices = market.get_current_prices(1).flatten()
        owned_stocks = np.where(self.portfolio > 0)[0]
        value = 0
        for i in range(len(owned_stocks)):
            value += self.portfolio[owned_stocks[i]] * \
            current_prices[owned_stocks[i]]
        value += self.balance
        return value

class CustomTrader(object):
    
    def __init__(self, balance, num_stocks, num_positions,
                 buy_filters, buy_ranking, sell_filters, emergency_sell):
        self.balance = balance
        self.portfolio = np.zeros(num_stocks)
        self.price_highs = np.zeros(num_stocks)
        self.num_stocks = num_stocks
        self.num_positions = num_positions
        self.open_positions = num_positions
        self.buy_filters = buy_filters
        self.buy_ranking = buy_ranking
        self.sell_filters = sell_filters
        self.emergency_sell = emergency_sell
        self.history = 0
        self.trades = 0
        for i in range(len(self.buy_filters)):
            if self.buy_filters[i].history > self.history:
                self.history = self.buy_filters[i].history
        if self.buy_ranking.history > self.history:
            self.history = self.buy_ranking.history
        for i in range(len(self.sell_filters)):
            if self.sell_filters[i].history > self.history:
                self.history = self.sell_filters[i].history
    
    def buy_stock(self, stock_number, payment, market):
        price = market.get_current_prices(1).flatten()[stock_number]
        charge = market.transaction_cost
        if self.balance < charge:
            payment = 0
        elif self.balance < (payment - charge):
            payment = self.balance
        self.balance -= payment
        self.portfolio[stock_number] += ((payment - charge) / price)
        self.price_highs[stock_number] = price
        self.trades += 1
    
    def sell_stock(self, stock_number, amount, market):
        price = market.get_current_prices(1).flatten()[stock_number]
        charge = market.transaction_cost
        if amount > self.portfolio[stock_number]:
            amount = self.portfolio[stock_number]
        self.balance += (amount * price) - charge
        self.portfolio[stock_number] -= amount
    
    def update_portfolio(self, market):
        current_prices = market.get_current_prices(self.history)
        owned_stocks = np.where(self.portfolio > 0)[0]
        self.price_highs[owned_stocks] = np.maximum(current_prices[-1,owned_stocks],
                                                    self.price_highs[owned_stocks])
        available_stocks = np.where(self.portfolio == 0)[0]
        if self.open_positions < self.num_positions:
            owned_stock_price_highs = self.price_highs[owned_stocks]
            owned_stock_prices = current_prices[-1,owned_stocks]
            decisions = np.ones(len(owned_stocks))
            for i in range(len(self.sell_filters)):
                decisions *= self.sell_filters[i](current_prices)[owned_stocks]
            emergency_decisions = (owned_stock_prices < self.emergency_sell * \
                                   owned_stock_price_highs).astype(np.int)
            decisions += emergency_decisions
            for i in range(len(owned_stocks)):
                if decisions[i] > 0:
                    self.sell_stock(owned_stocks[i],
                                    self.portfolio[owned_stocks[i]],
                                    market)
                    self.open_positions += 1
        
        decisions = np.ones(len(available_stocks))
        for i in range(len(self.buy_filters)):
            decisions *= self.buy_filters[i](current_prices)[available_stocks]
        decisions *= self.buy_ranking(current_prices)[available_stocks]
        order = np.argsort(decisions)[::-1]
        for i in range(self.open_positions):
            if decisions[order[i]] > 0:
                self.buy_stock(available_stocks[order[i]],
                               self.balance / self.open_positions,
                               market)
                self.open_positions -= 1
            else:
                break
        
    def get_value(self, market):
        current_prices = market.get_current_prices(1).flatten()
        owned_stocks = np.where(self.portfolio > 0)[0]
        value = 0
        for i in range(len(owned_stocks)):
            value += self.portfolio[owned_stocks[i]] * \
            current_prices[owned_stocks[i]]
        value += self.balance
        return value