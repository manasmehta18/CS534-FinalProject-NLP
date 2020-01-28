import nltk
nltk.download('punkt')
sentence1 = "your report may, if you wish, include some images or video material."
sentence2 = "apple bottom jeans have cheese."
sentence_list = [sentence1, sentence2]

tokens = []
for sentence in sentence_list:
    tokens.append(nltk.word_tokenize(sentence))

print(tokens)

output2 = "[[0 15 5 (Xp) [2 4 3 (Xx) [2 4 3 (Xx)"

output = output2.replace('[', '').split()

index = 0
start = 0
end = 0
global_features = set()
description_features = set()
for token in tokens:

    for i in output:

        if index % 4 == 0:
            start = int(i)
        elif index % 4 == 1:
            end = int(i)
            print(end)
        elif index % 4 == 2:
            pass
        elif index % 4 == 3:
            link = i.replace('(',  '').replace(')', '')
            if start == 0:
                word1 = "left_wall"
                word2 = token[end - 1]
            else:
                word1 = token[start - 1]
                word2 = token[end - 1]

            list_of_features.add((word1, word2, link))
            print(list_of_features)
        index += 1

print(output)