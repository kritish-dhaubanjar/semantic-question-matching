from flask import Flask, request, render_template
import numpy as np
from fuzzywuzzy import fuzz
from gensim.models import KeyedVectors
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()

import pickle
classifier = pickle.load(open('ANN.model', 'rb'))
randomForest = pickle.load(open('RandomForest.model', 'rb'))
logistic = pickle.load(open('LogisticRegression.model', 'rb'))
knn = pickle.load(open('KNN.model', 'rb'))

def wordmoverdistance(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    s1 = [lemmatizer.lemmatize(w) for w in s1 if w not in stopwords.words('english')]
    s2 = [lemmatizer.lemmatize(w) for w in s2 if w not in stopwords.words('english')]
    return model.wmdistance(s1, s2)
    
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stopwords.words('english')]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

X_Scaler = pickle.load(open('XScaler','rb'))    
    
app = Flask(__name__)
model = KeyedVectors.load('word2vec.model')

@app.route("/")
def index():
    question1 = 'What practical applications might evolve from the discovery of the Higgs Boson ?'
    question2 = 'What are some practical benefits of discovery of the Higgs Boson ?'
    diff_len = len(str(question1)) - len(str(question2))
    common_words = len(set(str(question1).lower().split()).intersection(set(str(question2).lower().split())))
    fuzz_qratio = fuzz.QRatio(str(question1), str(question2))
    fuzz_WRatio = fuzz.WRatio(str(question1), str(question2))
    fuzz_partial_ratio = fuzz.partial_ratio(str(question1), str(question2))
    fuzz_partial_token_set_ratio = fuzz.partial_token_set_ratio(str(question1), str(question2))
    fuzz_partial_token_sort_ratio = fuzz.partial_token_sort_ratio(str(question1), str(question2))
    fuzz_token_set_ratio = fuzz.token_set_ratio(str(question1), str(question2))
    fuzz_token_sort_ratio = fuzz.token_sort_ratio(str(question1), str(question2))
    wmd = wordmoverdistance(question1, question2)
    
    question1_vectors = sent2vec(question1)
    question2_vectors = sent2vec(question2)
    
    cosine_distance = cosine(question1_vectors, question2_vectors)
    cityblock_distance = cityblock(question1_vectors, question2_vectors)
    canberra_distance = canberra(question1_vectors, question2_vectors)
    euclidean_distance = euclidean(question1_vectors, question2_vectors)
    minkowski_distance = minkowski(question1_vectors, question2_vectors, 3)
    braycurtis_distance = braycurtis(question1_vectors, question2_vectors)
    
    X = np.array([diff_len, common_words, fuzz_qratio, fuzz_WRatio, fuzz_partial_ratio, fuzz_partial_token_set_ratio, 
             fuzz_partial_token_sort_ratio, fuzz_token_set_ratio, fuzz_token_sort_ratio, wmd, cosine_distance,
             cityblock_distance, canberra_distance, euclidean_distance, minkowski_distance, braycurtis_distance
             ])
    print(X)
    X = X_Scaler.transform(X.reshape(1,-1))
    y_pred = classifier.predict(X)
    print(y_pred)
    return (y_pred > 0.5)

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    