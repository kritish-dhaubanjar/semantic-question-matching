import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import nltk
import re
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

df = pd.read_csv("quora_duplicate_questions.tsv", delimiter="\t")

df.drop(["id", "qid1", "qid2", "is_duplicate"], axis=1, inplace=True)

questions = df["question1"]
questions = questions.append(df["question2"])

questions.dropna(inplace=True)

corpus = []
for question in questions:
    question = re.sub("[^a-zA-Z]", " ", question)
    question = (question.lower()).split()
    question = [lemmatizer.lemmatize(word) for word in question if not word in stopwords.words("english")]
    corpus.append(question)
    
model = Word2Vec(corpus, min_count=5, size=100, workers=2)

import pickle as pickle
pickle.dump(model, open("quora_word2vec.model", "wb"))

quora_word2vec = pickle.load(open("quora_word2vec.model", "rb"))
quora_word2vec.similar_by_word("world")