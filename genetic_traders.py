import numpy as np
from copy import deepcopy
import markets
import sys

class MAFilter(object):
    
    def __init__(self, a=int(np.random.uniform(5, 201)),
                 b=int(np.random.uniform(5, 201)),
                 thresh=np.random.uniform(0.8, 1.2)):
        self.a = a
        self.b = b
        self.thresh = thresh
        self.history = max(a, b)
    
    def __call__(self, prices):
        a_avg = prices[-self.a:].mean(0)
        b_avg = prices[-self.b:].mean(0)
        result = b_avg / a_avg
        decision = (result > self.thresh).astype(np.int)
        return decision
    
    def mutate(self):
        self.a = np.clip(self.a + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.b = np.clip(self.b + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.01),
                         0.5, 2.0)
        self.history = self.history = max(self.a, self.b)
    
    def randomise(self):
        self.a = int(np.random.uniform(5, 201))
        self.b = int(np.random.uniform(5, 201))
        self.thresh = np.random.uniform(0.8, 1.2)
        self.history = self.history = max(self.a, self.b)

class MARegimeFilter(object):
    
    def __init__(self, a=int(np.random.uniform(5, 201)),
                 b=int(np.random.uniform(5, 201)),
                 thresh=np.random.uniform(0.8, 1.2)):
        self.a = a
        self.b = b
        self.thresh = thresh
        self.history = max(a, b)
    
    def __call__(self, prices):
        mean_prices = prices.mean(1)
        a_avg = mean_prices[-self.a:].mean(0)
        b_avg = mean_prices[-self.b:].mean(0)
        result = b_avg / a_avg
        if result > self.thresh:
            decision = np.ones(prices.shape[1], dtype=np.int)
        else:
            decision = np.zeros(prices.shape[1], dtype=np.int)
        return decision
    
    def mutate(self):
        self.a = np.clip(self.a + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.b = np.clip(self.b + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.01),
                         0.5, 2.0)
        self.history = self.history = max(self.a, self.b)
    
    def randomise(self):
        self.a = int(np.random.uniform(5, 201))
        self.b = int(np.random.uniform(5, 201))
        self.thresh = np.random.uniform(0.8, 1.2)
        self.history = self.history = max(self.a, self.b)

class MaxPriceFilter(object):
    
    def __init__(self, thresh=np.random.uniform(0, 7)):
        self.thresh = thresh
        self.history = 1
    
    def __call__(self, prices):
        decision = (prices[-1] < np.exp(self.thresh)).astype(np.int)
        return decision
    
    def mutate(self):
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.05), 0, 7)
    
    def randomise(self):
        self.thresh = np.random.uniform(0, 7)

class MinPriceFilter(object):
    
    def __init__(self, thresh=np.random.uniform(0, 7)):
        self.thresh = thresh
        self.history = 1
    
    def __call__(self, prices):
        decision = (prices[-1] > np.exp(self.thresh)).astype(np.int)
        return decision
    
    def mutate(self):
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.05), 0, 7)
    
    def randomise(self):
        self.thresh = np.random.uniform(0, 7)

class ROCMinMeanFilter(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201)),
                 thresh=np.random.uniform(-10, 10)):
        self.n = n
        self.thresh = thresh
        self.history = n
    
    def __call__(self, prices):
        roc = (100 * (prices[-self.n+1:] / prices[-self.n:-1] - 1)).mean(0)
        decision = (roc > self.thresh).astype(np.int)
        return decision
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.05),
                         -20.0, 20.0)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.thresh = np.random.uniform(-10, 10)
        self.history = self.n

class ROCMinMeanRegimeFilter(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201)),
                 thresh=np.random.uniform(-10, 10)):
        self.n = n
        self.thresh = thresh
        self.history = n
    
    def __call__(self, prices):
        mean_prices = prices.mean(1)
        roc = (100 * (mean_prices[-self.n+1:] / mean_prices[-self.n:-1] - 1)).mean()
        if roc > self.thresh:
            decision = np.ones(prices.shape[1], dtype=np.int)
        else:
            decision = np.zeros(prices.shape[1], dtype=np.int)
        return decision
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.05),
                         -20.0, 20.0)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.thresh = np.random.uniform(-10, 10)
        self.history = self.n

class ROCMaxMeanFilter(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201)),
                 thresh=np.random.uniform(-10, 10)):
        self.n = n
        self.thresh = thresh
        self.history = n
    
    def __call__(self, prices):
        roc = (100 * (prices[-self.n+1:] / prices[-self.n:-1] - 1)).mean(0)
        decision = (roc < self.thresh).astype(np.int)
        return decision
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.05),
                         -20.0, 20.0)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.thresh = np.random.uniform(-10, 10)
        self.history = self.n

class ROCMaxMeanRegimeFilter(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201)),
                 thresh=np.random.uniform(-10, 10)):
        self.n = n
        self.thresh = thresh
        self.history = n
    
    def __call__(self, prices):
        mean_prices = prices.mean(1)
        roc = (100 * (mean_prices[-self.n+1:] / mean_prices[-self.n:-1] - 1)).mean()
        if roc < self.thresh:
            decision = np.ones(prices.shape[1], dtype=np.int)
        else:
            decision = np.zeros(prices.shape[1], dtype=np.int)
        return decision
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.05),
                         -20.0, 20.0)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.thresh = np.random.uniform(-10, 10)
        self.history = self.n

class ROCMinStdFilter(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201)),
                 thresh=np.random.uniform(0, 10)):
        self.n = n
        self.thresh = thresh
        self.history = n
    
    def __call__(self, prices):
        roc = (100 * (prices[-self.n+1:] / prices[-self.n:-1] - 1)).std(0)
        decision = (roc > self.thresh).astype(np.int)
        return decision
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.05),
                         -20.0, 20.0)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.thresh = np.random.uniform(-10, 10)
        self.history = self.n

class ROCMinStdRegimeFilter(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201)),
                 thresh=np.random.uniform(-10, 10)):
        self.n = n
        self.thresh = thresh
        self.history = n
    
    def __call__(self, prices):
        mean_prices = prices.mean(1)
        roc = (100 * (mean_prices[-self.n+1:] / mean_prices[-self.n:-1] - 1)).std()
        if roc > self.thresh:
            decision = np.ones(prices.shape[1], dtype=np.int)
        else:
            decision = np.zeros(prices.shape[1], dtype=np.int)
        return decision
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.05),
                         -20.0, 20.0)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.thresh = np.random.uniform(-10, 10)
        self.history = self.n

class ROCMaxStdFilter(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201)),
                 thresh=np.random.uniform(0, 10)):
        self.n = n
        self.thresh = thresh
        self.history = n
    
    def __call__(self, prices):
        roc = (100 * (prices[-self.n+1:] / prices[-self.n:-1] - 1)).std(0)
        decision = (roc < self.thresh).astype(np.int)
        return decision
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.05),
                         -20.0, 20.0)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.thresh = np.random.uniform(-10, 10)
        self.history = self.n

class ROCMaxStdRegimeFilter(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201)),
                 thresh=np.random.uniform(-10, 10)):
        self.n = n
        self.thresh = thresh
        self.history = n
    
    def __call__(self, prices):
        mean_prices = prices.mean(1)
        roc = (100 * (mean_prices[-self.n+1:] / mean_prices[-self.n:-1] - 1)).std()
        if roc < self.thresh:
            decision = np.ones(prices.shape[1], dtype=np.int)
        else:
            decision = np.zeros(prices.shape[1], dtype=np.int)
        return decision
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.thresh = np.clip(self.thresh + np.random.normal(0, 0.05),
                         -20.0, 20.0)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.thresh = np.random.uniform(-10, 10)
        self.history = self.n

class MARanking(object):
    
    def __init__(self, a=int(np.random.uniform(5, 201)),
                 b=int(np.random.uniform(5, 201))):
        self.a = a
        self.b = b
        self.history = max(a, b)
    
    def __call__(self, prices):
        a_avg = prices[-self.a:].mean(0)
        b_avg = prices[-self.b:].mean(0)
        ranking = b_avg / a_avg
        return ranking
    
    def mutate(self):
        self.a = np.clip(self.a + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.b = np.clip(self.b + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.history = self.history = max(self.a, self.b)
    
    def randomise(self):
        self.a = int(np.random.uniform(5, 201))
        self.b = int(np.random.uniform(5, 201))
        self.history = self.history = max(self.a, self.b)

class ROCMaxMeanRanking(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201))):
        self.n = n
        self.history = n
    
    def __call__(self, prices):
        ranking = (100 * (prices[-self.n+1:] / prices[-self.n:-1] - 1)).mean(0)
        return ranking
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.history = self.n

class ROCMinMeanRanking(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201))):
        self.n = n
        self.history = n
    
    def __call__(self, prices):
        ranking = -1*(100 * (prices[-self.n+1:] / prices[-self.n:-1] - 1)).mean(0)
        return ranking
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.history = self.n

class ROCMaxStdRanking(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201))):
        self.n = n
        self.history = n
    
    def __call__(self, prices):
        ranking = (100 * (prices[-self.n+1:] / prices[-self.n:-1] - 1)).std(0)
        return ranking
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.history = self.n

class ROCMinStdRanking(object):
    
    def __init__(self, n=int(np.random.uniform(5, 201))):
        self.n = n
        self.history = n
    
    def __call__(self, prices):
        ranking = -1*(100 * (prices[-self.n+1:] / prices[-self.n:-1] - 1)).std(0)
        return ranking
    
    def mutate(self):
        self.n = np.clip(self.n + np.random.choice([-5,-4,-3,-2,-1,1,2,3,4,5]),
                         5, 200)
        self.history = self.n
    
    def randomise(self):
        self.n = int(np.random.uniform(5, 201))
        self.history = self.n

class TraderGenome(object):
    
    def __init__(self, buy_filters, buy_ranking, sell_filters, emergency_sell):
        self.buy_filters = buy_filters
        self.buy_ranking = buy_ranking
        self.sell_filters = sell_filters
        self.emergency_sell = emergency_sell
    
    def mutate(self, pop_buy_filters, pop_buy_rankings,
               pop_sell_filters, delete_filter_prob,
               add_filter_prob, replace_ranking_prob,
               mutation_prob, randomise_prob):
        if np.random.uniform(0, 1) < delete_filter_prob \
        and len(self.buy_filters) > 1:
            index = np.random.choice(len(self.buy_filters))
            del self.buy_filters[index]
        
        if np.random.uniform(0, 1) < add_filter_prob:
            buy_filter = np.random.choice(pop_buy_filters)()
            self.buy_filters.append(buy_filter)
        
        if np.random.uniform(0, 1) < delete_filter_prob \
        and len(self.sell_filters) > 1:
            index = np.random.choice(len(self.sell_filters))
            del self.sell_filters[index]
        
        if np.random.uniform(0, 1) < add_filter_prob:
            sell_filter = np.random.choice(pop_sell_filters)()
            self.sell_filters.append(sell_filter)
        
        if np.random.uniform(0, 1) < replace_ranking_prob:
            self.buy_ranking = np.random.choice(pop_buy_rankings)()
        
        for i in range(len(self.buy_filters)):
            if np.random.uniform(0, 1) < randomise_prob:
                self.buy_filters[i].randomise()
            elif np.random.uniform(0, 1) < mutation_prob:
                self.buy_filters[i].mutate()
        
        for i in range(len(self.sell_filters)):
            if np.random.uniform(0, 1) < randomise_prob:
                self.sell_filters[i].randomise()
            elif np.random.uniform(0, 1) < mutation_prob:
                self.sell_filters[i].mutate()
        
        if np.random.uniform(0, 1) < randomise_prob:
            self.buy_ranking.randomise()
        elif np.random.uniform(0, 1) < mutation_prob:
            self.buy_ranking.mutate()
        
        if np.random.uniform(0, 1) < randomise_prob:
            self.emergency_sell = np.random.uniform(0, 1)
        if np.random.uniform(0, 1) < mutation_prob:
            self.emergency_sell = np.clip(self.emergency_sell + \
                                          np.random.normal(0, 0.01),
                                          0, 1)
    
    def partial_crossover(self, other):
        if np.random.uniform(0, 1) < 0.5:
            buy_filters = deepcopy(self.buy_filters)
        else:
            buy_filters = deepcopy(other.buy_filters)
        
        if np.random.uniform(0, 1) < 0.5:
            buy_ranking = deepcopy(self.buy_ranking)
        else:
            buy_ranking = deepcopy(other.buy_ranking)
        
        if np.random.uniform(0, 1) < 0.5:
            sell_filters = deepcopy(self.sell_filters)
        else:
            sell_filters = deepcopy(other.sell_filters)
        
        if np.random.uniform(0, 1) < 0.5:
            emergency_sell = deepcopy(self.emergency_sell)
        else:
            emergency_sell = deepcopy(other.emergency_sell)
        
        child = TraderGenome(buy_filters, buy_ranking,
                             sell_filters, emergency_sell)
        return child
    
    def full_crossover(self, other):
        all_buy_filters = self.buy_filters + other.buy_filters
        n_samples = min(len(self.buy_filters), len(other.buy_filters))
        indices = np.random.choice(len(all_buy_filters), n_samples)
        buy_filters = []
        for i in range(n_samples):
            buy_filters.append(deepcopy(all_buy_filters[indices[i]]))
        
        if np.random.uniform(0, 1) < 0.5:
            buy_ranking = deepcopy(self.buy_ranking)
        else:
            buy_ranking = deepcopy(other.buy_ranking)
        
        all_sell_filters = self.sell_filters + other.sell_filters
        n_samples = min(len(self.sell_filters), len(other.sell_filters))
        indices = np.random.choice(len(all_sell_filters), n_samples)
        sell_filters = []
        for i in range(n_samples):
            sell_filters.append(deepcopy(all_sell_filters[indices[i]]))
        
        if np.random.uniform(0, 1) < 0.5:
            emergency_sell = deepcopy(self.emergency_sell)
        else:
            emergency_sell = deepcopy(other.emergency_sell)
        
        child = TraderGenome(buy_filters, buy_ranking,
                             sell_filters, emergency_sell)
        return child

class GeneticTrader(object):
    
    def __init__(self, balance, num_stocks, num_positions, genome):
        self.balance = balance
        self.portfolio = np.zeros(num_stocks)
        self.price_highs = np.zeros(num_stocks)
        self.num_stocks = num_stocks
        self.num_positions = num_positions
        self.open_positions = num_positions
        self.buy_filters = genome.buy_filters
        self.buy_ranking = genome.buy_ranking
        self.sell_filters = genome.sell_filters
        self.emergency_sell = genome.emergency_sell
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

class TraderPopulation(object):
    
    def __init__(self, pop_size, pop_buy_filters, pop_buy_rankings,
                 pop_sell_filters, delete_filter_prob,
                 add_filter_prob, replace_ranking_prob,
                 mutation_prob, randomise_prob):
        self.pop_size = pop_size
        self.pop_buy_filters = pop_buy_filters
        self.pop_buy_rankings = pop_buy_rankings
        self.pop_sell_filters = pop_sell_filters
        self.delete_filter_prob = delete_filter_prob
        self.add_filter_prob = add_filter_prob
        self.replace_ranking_prob = replace_ranking_prob
        self.mutation_prob = mutation_prob
        self.randomise_prob = randomise_prob
        self.pop = []
        for i in range(self.pop_size):
            buy_filters = [np.random.choice(self.pop_buy_filters)()]
            buy_ranking = np.random.choice(self.pop_buy_rankings)()
            sell_filters = [np.random.choice(self.pop_sell_filters)()]
            emergency_sell = np.random.uniform(0, 1)
            self.pop.append(TraderGenome(buy_filters, buy_ranking,
                                         sell_filters, emergency_sell))
        self.fitnesses = None
        self.fitness_history = None
    
    def fitness(self, balance, num_stocks, num_positions,
                transaction_cost, noise, runs, run_length):
        self.fitnesses = np.zeros(self.pop_size)
        for i in range(runs):
            start = int(np.random.uniform(200, 3927 - run_length))
            market = markets.NoisyMarket(start, num_stocks, transaction_cost, noise)
            for j in range(self.pop_size):
                trader = GeneticTrader(balance, num_stocks, num_positions,
                                       deepcopy(self.pop[j]))
                market.reset()
                for k in range(run_length):
                    trader.update_portfolio(market)
                    market.step()
                value = trader.get_value(market)
                self.fitnesses[j] += value/runs
    
    def selection(self):
        new_pop = []
        indices = np.argsort(self.fitnesses)[self.pop_size//2:]
        for i in range(indices.shape[0]):
            new_pop.append(deepcopy(self.pop[indices[i]]))
        self.pop = new_pop
    
    def reproduce(self):
        new_pop = []
        for i in range(self.pop_size):
            parent1, parent2 = np.random.choice(self.pop, 2, replace=False)
            if np.random.uniform(0, 1) < 0.5:
                child = parent1.partial_crossover(parent2)
            else:
                child = parent1.full_crossover(parent2)
            child.mutate(self.pop_buy_filters, self.pop_buy_rankings,
                         self.pop_sell_filters, self.delete_filter_prob,
                         self.add_filter_prob, self.replace_ranking_prob,
                         self.mutation_prob, self.randomise_prob)
            new_pop.append(child)
        self.pop = new_pop
    
    def evolve(self, generations, balance, num_stocks, num_positions,
               transaction_cost, noise, runs, run_length):
        if self.fitnesses is None:
            self.fitness(balance, num_stocks, num_positions,
                         transaction_cost, noise, runs, run_length)
            self.fitness_history = [self.fitnesses[:]]
        for i in range(generations):
            self.selection()
            self.reproduce()
            self.fitness(balance, num_stocks, num_positions,
                         transaction_cost, noise, runs, run_length)
            self.fitness_history.append(self.fitnesses[:])
            self.best = deepcopy(self.pop[np.argmax(self.fitnesses)])
            sys.stdout.write('\rGeneration: %d, Best: %f' % \
                             (i+1, self.fitnesses.max()))
            sys.stdout.flush()
    
    def get_best(self):
        return self.best
    
    def get_fitness_history(self):
        return np.array(self.fitness_history[:])