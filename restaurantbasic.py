import pandas as pd 
import pickle

df = pd.read_csv("Restaurant_Reviews.tsv", sep='\t')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []

for i in range(0,len(df)):
    review = re.sub('[^a-zA-Z]',' ',df['Review'][i])    
    review = review.lower()                              
    review = review.split()                              
    review = [ps.stem(word)  for word in review if word not in set(stopwords.words('english'))]      
    review = ' '.join(review)                            
    corpus.append(review)    

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()

x = tf.fit_transform(corpus).toarray()

y = df['Liked']

pickle.dump(tf, open('restf.pkl','wb'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import MultinomialNB
restmodel = MultinomialNB().fit(X_train, y_train)

pickle.dump(restmodel, open('resclassifier.pkl','wb'))