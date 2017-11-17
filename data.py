from nltk_preprocessor import NLTKPreprocessor
from w2v_model import Word2VecVectorizer
from os import path, listdir

def window(data, max_len):
    for i in range(len(data)):
        idx = i - max_len
        yield data[max(idx, 0):idx+max_len], data[idx+max_len:idx+max_len+1]

def get_data(input):
    tokenizer = NLTKPreprocessor(
        stopwords=[], 
        punct=[], 
        lower=True, 
        strip=True, 
        lemmatize=False, 
        ignore_type=[])
    data = tokenizer.transform(input)
    return data

BASE_PATH = path.dirname(__file__)
CORPUS_PATH = 'corpus/obama/'

def read_all_txt():
    content = ''
    corpus_dir = path.join(BASE_PATH, CORPUS_PATH)
    for file_name in listdir(corpus_dir):
        full_path = path.join(corpus_dir, file_name)
        if path.isfile(full_path):
            with open(full_path, 'r') as text_file:
                file_content = text_file.read()
            file_content = file_content.replace('<Applause.>', '')
            file_content = file_content.replace('\n', ' ')
            content += file_content
    return content

WINDOW_SIZE = 6
def get_train_data():
    vectorizer = Word2VecVectorizer(sent_size=WINDOW_SIZE)
    X_train = []
    y_train = []

    data = [read_all_txt()]
    data = get_data(data)[0]
    data = window(data, WINDOW_SIZE)

    for X, y in data:
        X_train.append(vectorizer.transform(X))
        y_train.append(vectorizer.transform_single(y[0]))

    return (X_train, y_train)