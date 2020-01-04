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


# replace Unicode apostrophies in text with normal apostrophies
def fix_unicode(text: str) -> str:
    return text.replace(u"\u2019","'")

# split the text into a sequence of words and periods 
# (so that we know where the sentence ends)

import re
from bs4 import BeautifulSoup
import requests

# get the text from a particular url
url = "https://www.oreilly.com/ideas/what-is-data-science"
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

# find post-radar-content div
content = soup.find("div", "post-radar-content") 
regex = r"[\w']+|[\.]" # matches a word or a period

document = []

# find unique words in each paragraph
for paragraph in content("p"):
    words = re.findall(regex, fix_unicode(paragraph.text))
    document.extend(words)

from collections import defaultdict
transitions =  defaultdict(list)

for prev, current in zip(document, document[1:]):
    transitions[prev].append(current)


def generate_using_bigrams() -> str:
    current = "." # this means the next word will start a sentence
    result = []
    while True:
        next_word_candidates =  transitions[current] # bigrams(current, _)
        current = random.choice(next_word_candidates) # choose one at random
        result.append(current) # append it to results
        if current == ".": return " ".join(result) # If "." we are done

trigram_transitions = defaultdict(list)
starts = []

for prev, current, next in zip(document, document[1:], document[2:]):
    if prev == ".": # if previous word was period
        starts.append(current) # then this is a start word
        
    trigram_transitions[(prev, current)].append(next)
    
def generate_using_trigrams() -> str:
    current = random.choice(starts) # choose a random starting word
    prev = "." # and precede it with a period
    result = [current]
    
    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next_word = random.choice(next_word_candidates)
        
        prev, current = current, next_word
        result.append(current)
        
        if current == ".":
            return " ".join(result)

from typing import List, Dict

# Type alias to refer to grammars later
Grammar = Dict[str, List[str]]

grammar = {
    "_S"  : ["_NP _VP"],
    "_NP" : ["_N",
             "_A _NP _P _A _N"],
    "_VP" : ["_V",
             "_V _NP"],
    "_N"  : ["data science", "Python", "regression"],
    "_A"  : ["big", "linear", "logistic"],
    "_P"  : ["about", "near"],
    "_V"  : ["learns", "trains", "tests", "is"]
}

def is_terminal(token: str) -> bool:
    return token[0] != "_"

def expand(grammar: Grammar, tokens: List[str]) -> List[str]:
    for i, token in enumerate(tokens):
        # If this is a terminal token, skip it.
        if is_terminal(token): continue

        # Otherwise, it's a non-terminal token,
        # so we need to choose a replacement at random.
        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            # Replacement could be e.g. "_NP _VP", so we need to
            # split it on spaces and splice it in.
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]

        # Now call expand on the new list of tokens.
        return expand(grammar, tokens)

    # If we get here we had all terminals and are done
    return tokens

# generate sentences
def generate_sentence(grammar: Grammar) -> List[str]:
    return expand(grammar, ["_S"])         
         
         
