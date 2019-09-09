"""
Created on Sun Sep  7 09:54:40 2019
@author: Neeraj
Description: This code performs basic hypothesis testing in Python. 
Reference: Chapter 7 : Hypothesis and Inference
"""
from typing import List, Dict
from collections import Counter
import math
import matplotlib.pyplot as plt

def bucketize(point: float, bucket_size: float) -> float:
    """Floor the point to the next lower multiple of bucket_size"""
    return bucket_size*math.floor(point/bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    """Buckets the points and counts how many are in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)
    
def plot_histogram(points: List[float], bucket_size: float, title: str):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width = bucket_size)
    plt.title(title)
    
# Let's use above functions to plot some histograms
import random
from probability import inverse_normal_cdf;

random.seed()
# sample 100 points uniformly between -100 and 100
uniform  = [200*random.random() - 100 for _ in range(100)]
# sample 100 points from normal distribution of mean 0 and std 57
normal = [57*inverse_normal_cdf(random.random()) for _ in range(10000)]

plot_histogram(uniform, 10, "Uniform Histogram")
plot_histogram(normal, 10, "Normal Distribution")

def random_normal() -> float:
    """Returns a random draw from standard normal distribution"""
    return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(1000)]
ys1 = [x + random_normal()/2 for x in xs]
ys2 = [-x + random_normal()/2  for x in xs]

plot_histogram(ys1, 0.5, "YS1")
plot_histogram(ys2, 0.5, "YS2")

plt.scatter(xs, ys1, marker = '.', color = 'red', label = 'ys1')
plt.scatter(xs, ys2, marker = '.', color = 'green', label = 'ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc = 9)
plt.title("Very Different Joint Distributions")
plt.show()

from statistics import correlation;
print(correlation(xs,ys1)) # about 0.9
print(correlation(xs,ys2)) # about -0.9

# Correlation matrix is a way to check pairwise correlations in 2 or more dimensions
from vector_operations import Vector; 
from matrix_operations import Matrix, make_matrix;

def correlation_matrix(data: List[Vector]) -> Matrix:
    """Creates a len(data) x len(data) matrix where (i-j)th element
    is correlation between data[i] and data[j]"""
    
    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])

    return make_matrix(len(data), len([data[0]]), correlation_ij)

# corr_data is a list of four 100-d vectors

# Just some random data to show off correlation scatterplots
num_points = 100

def random_row() -> List[float]:
    row = [0.0, 0, 0, 0]
    row[0] = random_normal()
    row[1] = -5 * row[0] + random_normal()
    row[2] = row[0] + row[1] + 5 * random_normal()
    row[3] = 6 if row[2] > -2 else 0
    return row

random.seed(0)
# each row has 4 points, but really we want the columns
corr_rows = [random_row() for _ in range(num_points)]

corr_data = [list(col) for col in zip(*corr_rows)]

num_vectors = len(corr_data)
fig, ax = plt.subplots(num_vectors, num_vectors)

for i in range(num_vectors):
    for j in range(num_vectors):
        # Scatter column_j on the x-axis vs. column on the y-axis
        if i != j: ax[i][j].scatter(corr_data[j], corr_data[i])
        # Unless i == j, in which case show the series name
        else: ax[i][j].annotate("series" + str(i), (0.5,0.5),
                               xycoords = 'axes fraction',
                               ha = "center", va = "center")
            
# Fix the bottom right and top left axis labels, which are wrong because
    # their charts only have text in them
ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())
#plt.show()

from collections import namedtuple
import datetime
StockPrice = namedtuple('StockPrice',['symbol', 'date', 'closing_price'])
price = StockPrice('MSFT', datetime.date(2018,12, 14), 106.3)

from typing import NamedTuple
class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float
        
    def is_high_tech(self) -> bool:
        return self.symbol in ['MSFT','GOOG','FB','AMZN','AAPL']
    
price = StockPrice('MSFT', datetime.date(2018,12, 14), 106.3)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.3
assert price.is_high_tech()

from typing import Optional
import re

def try_parse_row(row: List[str]) -> Optional[StockPrice]:
    symbol, date_, closing_price_ = row

    # Stock symbol should be all capital letters
    if not re.match(r"^[A-Z]+$", symbol):
        return None

    try:
        date = parse(date_).date()
    except ValueError:
        return None

    try:
        closing_price = float(closing_price_)
    except ValueError:
        return None

    return StockPrice(symbol, date, closing_price)

from dateutil.parser import parse
import csv

with open("stocks.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = [[row['Symbol'], row['Date'], row['Close']]
            for row in reader]

# skip header
maybe_data = [try_parse_row(row) for row in rows]

# Make sure they all loaded successfully:
assert maybe_data
assert all(sp is not None for sp in maybe_data)

# This is just to make mypy happy
data = [sp for sp in maybe_data if sp is not None]

max_aapl_price = max(stock_price.closing_price 
                     for stock_price in data 
                     if stock_price.symbol == 'AAPL')
print(max_aapl_price)

from collections import defaultdict

max_prices: Dict[str, float] = defaultdict(lambda: float('-inf'))
    
for sp in data:
    symbol, closing_price = sp.symbol, sp.closing_price
    
    if closing_price > max_prices[symbol]:
        max_prices[symbol] = closing_price
max_prices

from typing import List
prices: Dict[str, List[float]] = defaultdict(list)

for sp in data:
    prices[sp.symbol].append(sp)
# order (or sort) the prices by date
prices = {symbol: sorted(symbol_prices) 
          for symbol, symbol_prices in prices.items()}
#print(prices)

def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    return today.closing_price / yesterday.closing_price - 1

class DailyChange(NamedTuple):
    symbol: str
    date: datetime.date
    pct_change: float
    
def day_over_day_changes(prices: List[StockPrice]) -> List[DailyChange]:
    """Assumes prices are for once stock and are ordered"""
    return [DailyChange(symbol = today.symbol,
                           date = today.date,
                           pct_change = pct_change(yesterday, today))
            for yesterday, today in zip(prices, prices[1:])]

all_changes = [change for symbol_prices in prices.values()
              for change in day_over_day_changes(symbol_prices)]
max_change = max(all_changes, key = lambda change: change.pct_change)

changes_by_month: List[DailyChange] = {month: [] for month in range(1,13)}
    
for change in all_changes:
    changes_by_month[change.date.month].append(change)

avg_daily_change = {month: sum(change.pct_change for change in changes)/ len(changes)
                   for month, changes in changes_by_month.items()}

assert avg_daily_change[10] == max(avg_daily_change.values())

from typing import Tuple
from vector_operations import vector_mean
from statistics import standard_deviation

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """Returns mean and standard deviation of each feature"""
    dim = len(data[0])
    means = vector_mean(data)
    stdevs = [standard_deviation([vector[i] for vector in data])
             for i in range(dim)]
    return means, stdevs

def rescale(data: List[Vector]) -> List[Vector]:
    """Rescales the data so that each feature has
    mean 0 and standard deviation 1, leaves the
    feature which has 0 standard deviation."""
    dim = len(data[0])
    means, stdevs = scale(data)
    
    # Make a copy of each vector
    rescaled = [v[:] for v in data]
    
    for v in rescaled:
        for i in range(dim):
            v[i] = (v[i] - means[i])/ stdevs[i]
            
    return rescaled

import tqdm

for i in tqdm.tqdm(range(100)):
    _ = [random.random for _ in range(10000000)]
    
from typing import List

def primes_up_to(n: int) -> List[int]:
    primes = [2]
    
    with tqdm.trange(3,n) as t:
        for i in t:
            # i is prime if no smaller prime divides it
            i_is_prime = not any(i % p == 0 for p in primes)
            if i_is_prime:
                primes.append(i)
                
            t.set_description(f"{len(primes)} primes")
    return primes

my_primes = primes_up_to(1000)
