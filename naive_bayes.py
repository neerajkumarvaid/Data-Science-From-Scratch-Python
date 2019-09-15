
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
