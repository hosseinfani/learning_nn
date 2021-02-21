import pandas as pd
import nltk
nltk.download('punkt')
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import params

df = pd.read_csv('./data/reddit_worldnews.csv')
#df['preprocess_title'] = df.apply(lambda row: simple_preprocess(row['title']), axis = 1)
#df.loc[:, 'preprocess_title'].to_csv('./data/preprocess_title', encoding='utf-8', index=False)
news_titles = df["title"].values

stop_words = stopwords.words('english')
def preprocess(text):
     text = text.lower()  # To lower
     doc = word_tokenize(text)  # Tokenize to words
     doc = [word for word in doc if word not in stop_words]  # Remove stopwords.
     doc = [word for word in doc if word.isalpha()]  # Remove numbers and special characters
     return doc

news_title_preprocess = [preprocess(title) for title in news_titles]
with open('./data/news_title_preprocess.txt', mode='w', encoding='utf-8') as f:
    for s in news_title_preprocess:
        while(len(s) < params.lm['w'] + 1):#we have to extend our titles to at least have w+1 words for our lm input+output context!
            s.insert(0, '<s>')

        f.write(' '.join(s))
        f.write('\n')