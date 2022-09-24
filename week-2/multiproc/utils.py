import re
from pathlib import Path

import pandas as pd
from joblib import load

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

PATH_DATA = Path(__file__).parent / 'data'

trained_models = Path(__file__).parent / 'trained_models/'
tfidf_vect = load(trained_models  / 'tfidf.joblib')
model = load(trained_models / 'model.joblib')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z +_-]')
STOPWORDS = stopwords.words('english')
RANDOM_STATE = 17


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    if pd.isna(text):
        return ''

    text = text.strip().lower()
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, '', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub('-', ' ', text)  # if there is defise split the words
    text = ' '.join([s for s in text.split() if s not in STOPWORDS])  # delete stopwords from text
    text = [token for token in text.split() if not token.isdigit()]  # remove digits
    text = ' '.join([s for s in text])

    return text


def prepare_input(multiplier=1):
    data = pd.read_csv(PATH_DATA / 'train.csv')
    data['text_prepared'] = data['excerpt'].apply(text_prepare)

    x = [t for t in data['text_prepared']] * multiplier
    return x


def predict(data):
    x = tfidf_vect.transform(data)
    pred = model.predict(x)

    return list(zip(data, pred))


if __name__ == '__main__':
    a = predict(prepare_input())
    print(a[:10])
