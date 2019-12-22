import pickle
import numpy as np

class Market(object):
    
    def __init__(self, start, num_stocks, transaction_cost):
        self.start = start
        self.time = start
        with open('daily_prices.pickle', 'rb') as handle:
            stock_prices = pickle.load(handle)
        stock_indices = np.random.choice(388, num_stocks, replace=False)
        self.stock_prices = stock_prices[:, stock_indices]
        self.transaction_cost = transaction_cost
    
    def reset(self):
        self.time = self.start
    
    def step(self):
        self.time += 1
    
    def get_current_prices(self, timesteps):
        prices = self.stock_prices[self.time+1-timesteps:self.time+1]
        return prices

class NoisyMarket(object):
    
    def __init__(self, start, num_stocks, transaction_cost, noise):
        self.start = start
        self.time = start
        with open('daily_prices.pickle', 'rb') as handle:
            stock_prices = pickle.load(handle)
        stock_indices = np.random.choice(388, num_stocks, replace=False)
        self.stock_prices = stock_prices[:, stock_indices]
        self.stock_prices *= np.random.uniform(1-noise/2, 1+noise/2,
                                               self.stock_prices.shape)
        self.transaction_cost = transaction_cost
    
    def reset(self):
        self.time = self.start
    
    def step(self):
        self.time += 1
    
    def get_current_prices(self, timesteps):
        prices = self.stock_prices[self.time+1-timesteps:self.time+1]
        return prices