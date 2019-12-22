import genetic_traders as gt
import numpy as np
import matplotlib.pyplot as plt

pop_size = 50
pop_buy_filters = [gt.MAFilter,
                   gt.MARegimeFilter,
                   gt.MaxPriceFilter,
                   gt.MinPriceFilter,
                   gt.ROCMinMeanFilter,
                   gt.ROCMaxMeanFilter]
pop_buy_rankings = [gt.ROCMaxMeanRanking,
                    gt.ROCMinMeanRanking,
                    gt.MARanking]
pop_sell_filters = [gt.MAFilter,
                    gt.MARegimeFilter,
                    gt.MaxPriceFilter,
                    gt.MinPriceFilter,
                    gt.ROCMinMeanFilter,
                    gt.ROCMaxMeanFilter]
delete_filter_prob = 0.1
add_filter_prob = 0.1
replace_ranking_prob = 0.1
mutation_prob = 0.5
randomise_prob = 0.05

generations = 50
balance = 1000
num_stocks = 200
num_positions = 20
transaction_cost = 0
noise = 0.1
runs = 5
run_length = 500

population = gt.TraderPopulation(pop_size, pop_buy_filters, pop_buy_rankings,
                                 pop_sell_filters, delete_filter_prob,
                                 add_filter_prob, replace_ranking_prob,
                                 mutation_prob, randomise_prob)
population.evolve(generations, balance, num_stocks, num_positions,
                  transaction_cost, noise, runs, run_length)
history = population.get_fitness_history()

fig, ax = plt.subplots(figsize=(12,8), nrows=1, ncols=1)
ax.scatter(np.arange(history.flatten().shape[0]),
           history.flatten(), alpha=0.5)
ax.set_yscale('log')
plt.show()