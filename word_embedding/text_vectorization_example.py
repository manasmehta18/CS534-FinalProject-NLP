import numpy as np
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

word_list = ['students', 'elect', 'follow', 'either', 'children', 'families', 'adult', 'services', 'pathway', 'module', 'although', 'sessions', 'delivered', 'jointly', 'students', 'draw', 'practice', 'experiences', 'date', 'consider', 'complexities', 'contemporary', 'social', 'work', 'practice', 'module', 'build', 'learning', 'previous', 'modules', 'enable', 'students', 'consider', 'advanced', 'law', 'theory']
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
print(mean_vec)
