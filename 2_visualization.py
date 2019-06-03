# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 00:55:07 2019

@author: Neeraj

Description: This code illustrates how to use common plotting methods such as line plots, bar graphs, scatter plots in Python.
Reference: Chapter 3 : Data Visualization 
"""

from matplotlib import pyplot as plt


"""Line chart"""

# Example 1
years = [1950,1960,1970,1980,1990,2000,2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# plot years on x-axis and gdp on y-axis using a line plot
plt.plot(years, gdp, color = 'green', marker = 'o', linestyle = 'solid')

# add title 
plt.title("Nominal GDP")
# add x-label and y-label
plt.xlabel("Value")
plt.ylabel("Years")

# Example 2
variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]

total_error = [x + y for x,y in zip(variance, bias_squared)]

xs = [i for i,_ in enumerate(variance)]

plt.plot(xs, variance, 'r-', label = 'Variance')
plt.plot(xs, bias_squared, 'g-.', label = "Bias^2")
plt.plot(xs, total_error, 'b:', label = "Total Error")
plt.legend(loc=9)
plt.xlabel("Model Complexity")
plt.title("Bias-Variance Trade-Off")

""" Bar graphs """

# Example 1
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

xs = [i for i, _ in enumerate(movies)]

plt.bar(xs, num_oscars)

plt.title("My Favourite Movies")
plt.ylabel("# of Oscars")

plt.xticks(xs, movies)

# Example 2
grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]

# Create 10 bins (80-89 maps to 80)
decile = lambda grade: grade//10 * 10

import math
tertile = lambda grade: math.ceil(grade/100 * 10)

from collections import Counter
histogram = Counter([decile(grade) for grade in grades])

# Plot histogram using a bar chart with bar width = 8
plt.bar([x for x in histogram.keys()], histogram.values(), 8)
plt.axis([-5, 105, 0, 5])
plt.xlabel("Decile")
plt.ylabel("# of students")
plt.title("Mid-term exam marks distribution")


""" Scatter plots """
friends = [ 70,  65,  72,  63,  71,  64,  60,  64,  67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels =  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends,minutes)

# annotate each point with friend's identity
for label, friend, minute_count in zip(labels, friends, minutes):
    plt.annotate(label, 
                 # provide x-y coordinates for annotation
                xy = [friend, minute_count],
                # offset the coordinates slightly to avoid label and point overlap 
                xytext = (5,-5),
                textcoords = "offset points")
plt.title("Daily minutes vs. number of friends")
plt.xlabel("# of friends")
plt.ylabel("Minutes spent on the site daily")















