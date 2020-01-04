# -*- coding: utf-8 -*-
"""
Created on Friday Jan  03 07:11:33 2020
@author: Neeraj
Description: This file contains an impelementation of natural language processing algorithms including n-grams, gibbs sampling, 
topic modeling, etc. from scratch in Python. It also contains code to generate word clouds.
Reference: Chapter 21 Natural Language Processing
"""
data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
         ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
         ("actionable insights", 40, 30), ("think out of the box", 45, 10),
         ("self-starter", 30, 50), ("customer focus", 65, 15),
         ("thought leadership", 35, 35)]

# How to creat a word cloud?

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 

stopwords = set(STOPWORDS) 

words = [t[0] for t in data]

# convert the list of words into a combined string
comment_words = ' '
for word in words:
    comment_words = comment_words + word + ' '

# create the word cloud
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words)

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0)

def text_size(total: int) -> float:
    """equals 8 if total is 0, 28 if total is 200"""
    return 8 + total/ 200 * 20

for word, job_popularity, resume_popularity in data:
    plt.text(job_popularity, resume_popularity, word,
            ha = 'center', va = 'center',
            size = text_size(job_popularity + resume_popularity))
plt.xlabel("Popularity on job postings")
plt.ylabel("Popularity on resumes")
plt.axis([0, 100, 0, 100])
plt.xticks([])
plt.yticks([])
plt.show()
