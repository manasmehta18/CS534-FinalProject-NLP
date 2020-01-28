import gensim.downloader as api
wv = api.load('word2vec-google-news-300')

# for i, word in enumerate(wv.vocab):
#     if i == 10:
#         break
#     print(word)

vec_king = wv['king']
print(type(vec_king))
print(vec_king)
vec_cameroon = wv['cameroon']
print(vec_cameroon)
