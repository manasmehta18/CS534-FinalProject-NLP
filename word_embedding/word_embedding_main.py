import json
import re
import csv
from preprocessing import tokenization
import numpy as np
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

def text_vectorization(word_list):
    vec_example = wv['king']
    vec_sum = np.zeros(vec_example.size)
    for word in word_list:
        try:
            word_vec = wv[word]
        except:
            word_vec = np.zeros(vec_example.size)

        vec_sum = vec_sum + word_vec

    if len(word_list) > 0:
        mean_vec = vec_sum/(len(word_list))
    else:
        mean_vec = np.zeros(vec_example.size)

    return mean_vec

regex1 = re.compile('(i\.e\.)')
regex2 = re.compile('(e\.g\.|e\.g)')
regex3 = re.compile('(etc|etc.)')

with open("../test.json", 'r') as file:
    data = json.load(file)
    for element in data['table']:
        course_description = element['description']
        description = regex3.sub('', regex2.sub('for example', regex1.sub('in other words', course_description)))
        word_list = tokenization(description)
        course_vector = text_vectorization(word_list)
        data_array = np.append(course_vector,[int(element['label'])])
        data_list = data_array.tolist()
        with open(r'vectorized_data.csv', mode='a') as course_vector_file:
            vector_writer = csv.writer(course_vector_file, delimiter=',')
            vector_writer.writerow(data_list)
