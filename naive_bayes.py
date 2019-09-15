"""
Created on Sun Sep 15 02:03:39 2019

@author: Neeraj

Description: Implemention of Naive Bayes' algorithm from scratch in Python.

Reference: Chapter 12: Naive Bayes Algorithm
"""

from typing import Set
import re

def tokenize(text: str) -> Set[str]:
    text = text.lower() # lowercase all the letter in text
    all_words = re.findall("[a-z0-9']+", text) # find all words containing alphabets, number and apostrophies
    # remove duplicates / extract unique words
    return set(all_words)

print(tokenize('Data Science is a science.'))

from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool

from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool

from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool

from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool

from typing import NamedTuple

class Message(NamedTuple):
    text: str
    is_spam: bool

from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict
class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k # smoothing factor
        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.ham_messages = self.spam_messages = 0
        
    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
            # increment word counts
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilties(self, token:str) -> Tuple[float, float]:
        """Computes P[token/spam] and P[token/ham]"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]
    
        p_token_spam = (spam + self.k)/(self.spam_messages + 2*self.k)
        p_token_ham = (ham + self.k)/(self.ham_messages + 2*self.k)
    
        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text) # extract each word from incoming text
        # initialize spam and ham probability to zero
        log_prob_if_spam = log_prob_if_ham = 0.0 
        
        # Iterate through each word in our vocabulary
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilties(token)
            #print(token, prob_if_spam,prob_if_ham )
            # If *token* appears in the message
            # add the log probability of seeing it
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)
                
            # Otherwise add log probability of not seeing it
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)
            
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
            
        return prob_if_spam/(prob_if_spam + prob_if_ham)     

messages = [Message("spam rules", is_spam = True),
           Message("ham rules", is_spam = False),
           Message("hello ham", is_spam = False)]

model = NaiveBayesClassifier(k = 0.5)
model.train(messages)

# Let's check if our model got all tokens right
assert model.tokens  == {"spam","ham","rules","hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

text = "hello spam"
model.predict(text)

from io import BytesIO # So we can treat bytes as a file
import requests # To download files which
import tarfile # are in .tar.bz format

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"

FILES = ["20021010_easy_ham.tar.bz2",
        "20021010_hard_ham.tar.bz2",
        "20021010_spam.tar.bz2"]

OUTPUT_DIR = "spam_data"

for filename in FILES:
    # Use requests to get the file contents at each URL
    content = requests.get(f"{BASE_URL}/{filename}").content
    
    # Wrap the in-memory bytes so we can use them as a file
    fin = BytesIO(content)
    
    # And extract all the files to the output directory
    with tarfile.open(fileobj= fin, mode = 'r:bz2') as tf:
        tf.extractall(OUTPUT_DIR)
        
import glob, re

path = "spam_data/*/*"
data: List[Message] = []
    
# glob.glob returns every filename that matches the wildcarded path

for filename in glob.glob(path):
    is_spam = "ham" not in filename
    
    # There are some garbage characters in the emails; the errors = 'ignore'
    # skips them instead of raising an exception
    with open(filename, errors = 'ignore') as email_file:
        for line in email_file:
            if line.startswith("Subject:"):
                subject = line.lstrip("Subject: ")
                data.append(Message(subject, is_spam))
                break # done with this file

import random
from machine_learning import split_data;

random.seed(0)
train_messages, test_messages = split_data(data, 0.75)

model = NaiveBayesClassifier()
model.train(train_messages)

from collections import Counter

predictions = [(message, model.predict(message.text))
              for message in test_messages]

confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                          for message, spam_probability in predictions)
print(confusion_matrix)

def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    prob_if_spam, prob_if_ham = model._probabilties(token)
    return prob_if_spam/ (prob_if_spam + prob_if_ham)

words = sorted(model.tokens, key = lambda t: p_spam_given_token(t,model))
print("spammiest_words", words[-10:])
print("hammiest_words", words[:10])
