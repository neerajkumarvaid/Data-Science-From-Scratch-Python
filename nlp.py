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
         
# Gibbs Sampling

from typing import Tuple
import random

def roll_a_die() -> int:
    return random.choice([1, 2, 3, 4, 5, 6])

def direct_sample() -> Tuple[int, int]:
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2

def random_y_given_x(x: int) -> int:
    """equally likely to be x + 1, x + 2, ... , x + 6"""
    return x + roll_a_die()

def random_x_given_y(y: int) -> int:
    if y <= 7:
        # if the total is 7 or less, the first die is equally likely to be
        # 1, 2, ..., (total - 1)
        return random.randrange(1, y)
    else:
        # if the total is 7 or more, the first die is equally likely to be
        # (total - 6), (total - 5), ..., 6
        return random.randrange(y - 6, 7)
    
def gibbs_sample(num_iters: int = 100) -> Tuple[int, int]:
    x, y = 1, 2 # doesn't really matter
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y

def compare_distributions(num_samples: int = 1000) -> Dict[int, List[int]]:
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[direct_sample()][1] += 1
    return counts         

# Topic modeling
import random
from typing import List
from collections import Counter

def sample_from(weights: List[float]) -> int:
    """Returns i with probability weights[i]/sum(weights)"""
    total = sum(weights)
    rnd = total * random.random() # uniform between 0 and total
    for i, w in enumerate(weights):
        rnd -= w   # return the smallest i such that
        if rnd <= 0: return i # weight[0] +...+ weights[i] >= rnd
        
# Draw 1000 times and count
draws = Counter(sample_from([0.1, 0.1, 0.8]) for _ in range(1000))

print(f"draws[0] = {draws[0]}")
print(f"draws[1] = {draws[1]}")
print(f"draws[2] = {draws[2]}")

documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]


K = 4

# How many times each topic is assigned to each document
# a list of counters, one for each document
document_topic_counts = [Counter() for _ in documents]

# How many times each word is assigned to each topic
# a list of counters, one for each topic
topic_word_counts = [Counter() for _ in range(K)]

# The total number of words assigned to each topic
# a list of numbers one for each topic
topic_counts = [0 for _ in range(K)]

# The total number of words in each document
# a list of numbers, ones for each document
document_lengths = [len(documents) for document in documents]

# The number of distinct words
distinc_words = set(word for document in documents for word in document)

# The number of documents
D = len(documents)

def p_topic_given_document(topic: int, d: int, alpha: float = 0.1) -> float:
    """The fraction of words in document 'd' that are assigned to 'topic'(+ some smoothing)"""
    return (document_topic_counts[d][topic] + alpha)/(document_lengths[d] + K*alpha)

def p_word_given_topic(word: str, topic: int, beta: float = 0.1) -> float:
    """The fraction of words assigned to 'topic' that equals word (+ some smoothing)"""
    return (topic_word_counts[topic][word] + beta)/(topic_counts[topic] + K*beta)

def topic_weight(d: int, word: str, k: int) -> float:
    """Given a document and a word in that document, return the weight for the kth topic"""
    return p_word_given_topic(word, k)*p_topic_given_document(k,d)

def choose_new_topic(d: int, word: str) -> int:
    return sample_from([topic_weight(d, word, k) for k in range(K)])

random.seed(0)
document_topics = [[random.randrange(K) for word in document]
                  for document in documents]

for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1

import tqdm

for iter in tqdm.trange(1000):
    for d in range(D):
        for i, (word, topic) in enumerate(zip(documents[d], document_topics[d])):
            # remove this word/topic from the counts
            # so that it doesn't influence the weigts
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1
            
            # choose new topic based on the weights
            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic
            
            # and now add it back to the counts
            document_topic_counts[d][new_topic] += 1
            topic_word_counts[new_topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1

for k, word_counts in enumerate(topic_word_counts):
    for word, count in word_counts.most_common():
        if count > 0: 
            print(k, word, count)         

topic_names = ["Big data and programming languages",
              "Python and statistics",
              "databases",
              "machine learning"]                  

for document, topic_counts in zip(documents, document_topic_counts):
    print(document)
    for topic, count in topic_counts.most_common():
        if count > 0:
            print(topic_names[topic], count)

from vector_operations import dot, Vector
import math

def cosine_similarity(v1: Vector, v2: Vector) -> float:
    return dot(v1, v2)/math.sqrt(dot(v1,v1) * dot(v2,v2))

print(f"cosine_similarity([1.,1.,1.], [2.,2.,2.]) = {cosine_similarity([1.,1.,1.], [2.,2.,2.])}")
print(f"cosine_similarity([-1.,-1.], [2.,2.]) = {cosine_similarity([-1.,-1.], [2.,2.])}")
print(f"cosine_similarity([1.,0.], [0.,1.]) = {cosine_similarity([1.,0.], [0.,1.])}")                  

import random
colors = ["red", "green", "blue", "yellow", "black", ""]
nouns = ["bed", "car", "boat", "cat"]
verbs = ["is", "was", "seems"]
adverbs = ["very", "quite", "extremely", ""]
adjectives = ["slow", "fast", "soft", "hard"]

def make_sentence() -> str:
    return " ".join(["The",
                    random.choice(colors),
                    random.choice(nouns),
                    random.choice(verbs),
                    random.choice(adverbs),
                    random.choice(adjectives),
                    "."])

NUM_SENTENCES = 50
random.seed(0)

sentences = [make_sentence() for _ in range(NUM_SENTENCES)]

for sentence in sentences:
    print(sentence)

#from deep_learning import Tensor
Tensor = list
from typing import List
class Vocabulary:
    def __init__(self, words: List[str] = None) -> None:
        self.w2i: Dict[str, int] = {} # mapping word to word_id
        self.i2w: Dict[int, str] = {} # mapping word_id to word
        
        for word in (words or []): # if words were provided
            self.add(word) # add them
            
    @property
    def size(self) -> int:
        """how many words are in the vocabulary"""
        return len(self.w2i)
        
    def add(self, word: str) -> None:
        if word not in self.w2i: # If the word is new to us:
            word_id = len(self.w2i) # Find the next id
            self.w2i[word] = word_id  # Add to the word -> word_id map
            self.i2w[word_id] = word # Add to the word_id -> map word 
                
    def get_id(self, word: str) -> int:
        """return the id of the word (or None)"""
        return self.w2i.get(word)
        
    def get_word(self, word_id: int) -> str:
        return self.i2w.get(word_id)
        
    def one_hot_encode(self, word: str) -> Tensor:
        word_id = self.get_id(word)
        assert word_id is not None, f"unkown word {word}"
            
        return [1.0 if i == word_id else 0.0 for i in  range(self.size)]

vocab = Vocabulary(["a","b","c"])
print(f"vocab.size = {vocab.size}")
character = "b"
print(f"vocab.get_id(b) = {vocab.get_id(character)}")
print(f"vocab.one_hot_encode(b) = {vocab.one_hot_encode(character)}")
character = "z"
print(f"vocab.get_id(z) = {vocab.get_id(character)}")
vocab.add("z")
print(f"vocab.size = {vocab.size}")
print(f"vocab.get_id(z) = {vocab.get_id(character)}")
print(f"vocab.one_hot_encode(z) = {vocab.one_hot_encode(character)}")

import json

def save_vocab(vocab: Vocabulary, filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(vocab.w2i, f)  # Only need to save w2i
        
def load_vocab(filename: str) -> Vocabulary:
    vocab = Vocabulary()
    with open(filename) as f:
        # Load w2i and generate i2w from it
        vocab.w2i = json.load(f)
        vocab.i2w = {id: word for word, id in vocab.w2i.items()}


from typing import Iterable
from deep_learning import Layer, Tensor, random_tensor, zero_like

class Embedding(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # One vector of size embedding_dim for each desired embedding
        self.embeddings = random_tensor(num_embeddings, embedding_dim)
        self.grad = zero_like(self.embeddings)
        
        # save last input_id
        self.last_input_id = None
        
    def forward(self, input_id: int) -> Tensor:
        """Just select the embedding vector corresponding to the input_id"""
        self.input_id = input_id # remember for use in backpropagation
        return self.embeddings[input_id]

    def backward(self, gradient: Tensor) -> None:
    # Zero out the gradient corresponding to the last input.
    # This way cheaper than creating a new all-zero tensor each time.
        if self.last_input_id is not None:
            zero_row = [0 for _ in range(self.embedding_dim)]
            self.grad[self.last_input_id] = zero_row
        
        self.last_input_id = self.input_id
        self.grad[self.input_id] = gradient
    
    def params(self) -> Iterable[Tensor]:
        return [self.embeddings]
    
    def grads(self) -> Iterable[Tensor]:
        return [self.grad]
        
from typing import Tuple

class TextEmbedding(Embedding):
    def __init__(self, vocab: Vocabulary, embedding_dim: int) -> None:
        # Call the superclass constructer
        super().__init__(vocab.size, embedding_dim)
        
        # And hang on to vocab
        self.vocab = vocab
        
    def __getitem__(self, word: str) -> Tensor:
        word_id = self.vocab.get_id(word)
        
        if word_id is not None:
            return self.embeddings[word_id]
        else:
            return None
        
    def closest(self, word: str, n: int = 5) -> List[Tuple[float, str]]:
        """Return n closest words based on cosine similarity"""
        vector = self(word)
        # Compute pairs (similarity, other_word), and sort most similar first
        scores = [(cosine_similarity(vector, self.embeddings[i]), other_word)
                  for other_word, i in self.vocab.w2i.items()]
        scores.sort(reverse=True)
        return scores[:n]

import re
    
    # This is not a great regex, but it works on our data.
tokenized_sentences = [re.findall("[a-z]+|[.]", sentence.lower())
                           for sentence in sentences]
    
# Create a vocabulary (that is, a mapping word -> word_id) based on our text.
vocab = Vocabulary(word
                       for sentence_words in tokenized_sentences
                       for word in sentence_words)
    
from deep_learning import Tensor

def one_hot_encode(i: int, num_labels: int = 10) -> List[float]:
        return [1.0 if j == i else 0.0 for j in range(num_labels)]
    
inputs: List[int] = []
targets: List[Tensor] = []
    
for sentence in tokenized_sentences:
        for i, word in enumerate(sentence):          # For each word
            for j in [i - 2, i - 1, i + 1, i + 2]:   # take the nearby locations
                if 0 <= j < len(sentence):           # that aren't out of bounds
                    nearby_word = sentence[j]        # and get those words.
    
                    # Add an input that's the original word_id
                    inputs.append(vocab.get_id(word))
    
                    # Add a target that's the one-hot-encoded nearby word
                    targets.append(vocab.one_hot_encode(nearby_word))

# Model for learning word vectors
    
from scratch.deep_learning import Sequential, Linear
    
random.seed(0)
EMBEDDING_DIM = 5  # seems like a good size
    
# Define the embedding layer separately, so we can reference it.
embedding = TextEmbedding(vocab=vocab, embedding_dim=EMBEDDING_DIM)
    
model = Sequential([
        # Given a word (as a vector of word_ids), look up its embedding.
        embedding,
        # And use a linear layer to compute scores for "nearby words".
        Linear(input_dim=EMBEDDING_DIM, output_dim=vocab.size)
    ])
    
    
# Train the word vector model
    
from scratch.deep_learning import SoftmaxCrossEntropy, Momentum, GradientDescent
    
loss = SoftmaxCrossEntropy()
optimizer = GradientDescent(learning_rate=0.01)
    
for epoch in range(100):
    epoch_loss = 0.0
    for input, target in zip(inputs, targets):
        predicted = model.forward(input)
        epoch_loss += loss.loss(predicted, target)
        gradient = loss.gradient(predicted, target)
        model.backward(gradient)
        optimizer.step(model)
    print(epoch, epoch_loss)            # Print the loss
    #print(embedding.closest("black"))   # and also a few nearest words
    #print(embedding.closest("slow"))    # so we can see what's being
    #print(embedding.closest("car"))     # learned.                  

# Explore most similar words
    
pairs = [(cosine_similarity(embedding[w1], embedding[w2]), w1, w2)
             for w1 in vocab.w2i
             for w2 in vocab.w2i
             if w1 < w2]
pairs.sort(reverse=True)
print(pairs[:5])


# Plot word vectors
    
from working_with_data import pca, transform
import matplotlib.pyplot as plt
    
# Extract the first two principal components and transform the word vectors
components = pca(embedding.embeddings, 2)
transformed = transform(embedding.embeddings, components)
    
# Scatter the points (and make them white so they're "invisible")
fig, ax = plt.subplots()
ax.scatter(*zip(*transformed), marker='.', color='w')
    
# Add annotations for each word at its transformed location
for word, idx in vocab.w2i.items():
    ax.annotate(word, transformed[idx])
    
# And hide the axes
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


from deep_learning import tensor_apply, tanh

class SimpleRnn(Layer):
    """Just about the simplest possible recurrent layer."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.w = random_tensor(hidden_dim, input_dim, init = 'xavier')
        self.u = random_tensor(hidden_dim, hidden_dim, init = 'xavier')
        self.b = random_tensor(hidden_dim)
        
        self.reset_hidden_state()
        
    def reset_hidden_state(self) -> None:
        self.hidden = [0 for _ in range(self.hidden_dim)]
        
    def forward(self, input: Tensor) -> Tensor:
        self.input = input  # Save both input and previous 
        self.prev_hidden = self.hidden # hidden state to use in backprop
        
        a = [dot(self.w[h], input) + dot(self.u[h], self.hidden) + self.b[h]
            for h in range(self.hidden_dim)]
        
        self.hidden = tensor_apply(tanh, a) # Apply tanh activation
        return self.hidden # and return the result
    
    def backward(self, gradient: Tensor):
        # Backpropagate through the tanh
        a_grad = [gradient[h] * (1 - self.hidden[h] ** 2)
                  for h in range(self.hidden_dim)]

        # b has the same gradient as a
        self.b_grad = a_grad

        # Each w[h][i] is multiplied by input[i] and added to a[h],
        # so each w_grad[h][i] = a_grad[h] * input[i]
        self.w_grad = [[a_grad[h] * self.input[i]
                        for i in range(self.input_dim)]
                       for h in range(self.hidden_dim)]

        # Each u[h][h2] is multiplied by hidden[h2] and added to a[h],
        # so each u_grad[h][h2] = a_grad[h] * prev_hidden[h2]
        self.u_grad = [[a_grad[h] * self.prev_hidden[h2]
                        for h2 in range(self.hidden_dim)]
                       for h in range(self.hidden_dim)]

        # Each input[i] is multiplied by every w[h][i] and added to a[h],
        # so each input_grad[i] = sum(a_grad[h] * w[h][i] for h in ...)
        return [sum(a_grad[h] * self.w[h][i] for h in range(self.hidden_dim))
                for i in range(self.input_dim)]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.u, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.u_grad, self.b_grad]
