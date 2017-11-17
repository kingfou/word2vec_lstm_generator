from nltk_preprocessor import NLTKPreprocessor
from w2v_model import Word2VecVectorizer

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