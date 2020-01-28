import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def tokenization(input_str):
    input_str = str(input_str)
    lower_str = input_str.lower()
    numberless_str = re.sub(r'\d+','',lower_str)
    nopunc_str = numberless_str.translate(string.maketrans("",""),string.punctuation)
    spaceless_str = nopunc_str.strip()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(spaceless_str)
    result = [i for i in tokens if not i in stop_words]

    return result
