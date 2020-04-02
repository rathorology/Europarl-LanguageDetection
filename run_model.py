from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import pandas as pd
import re
import pickle

with open('count_vectorizer_label_encoder.pkl', 'rb') as f:
    count_vectorizer, label_encoder = pickle.load(f)

with open('model_MNB.pkl', 'rb') as f:
    clf = pickle.load(f)

test_fs = pd.read_csv('europarl.test', sep="\t", header=None)

def clean_sentence(sent):
    
    # Remove () part from sent and <> tags
    
    sent = re.sub("\([^)]*\)", '',sent)
    sent = re.sub("<[^>]*>", '', sent)
    
    # Split into multiple sentences if more than one sentence
    return sent #[x for x in sent.split('\n') if x != '']

# Clean sentence
test_fs[1] = [clean_sentence(i) for i in test_fs[1]]

# Run model on fellowship.ai test dataset
print("MNB ACC: ", f1_score(label_encoder.transform(test_fs[0]), clf.predict(count_vectorizer.transform(test_fs[1])), average='macro'))


