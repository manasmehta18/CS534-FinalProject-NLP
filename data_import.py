# Importing:
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Class for modules (course names, course codes, course descriptions, labels)
class Modules:

    # Can provide training set, testing set or all, can shuffle
    def __init__(self, subset = 'all', shuffle = True):
        self.title = list()
        self.code = list()
        self.description = list()
        self.label = list()

        # Open and read json file into lists:
        with open('test.json') as json_file:
            data = json.load(json_file)
            for p in data['table']:
                self.title.append(p['title'])
                self.code.append(p['code'])
                self.description.append(p['description'])
                self.label.append(p['label'])

        # Shuffle on request:
        if shuffle == True:
            self.shuffle_data()
        else:
            pass

        # Data seperation:
        if subset == 'train':
            self.seperate_data(0.8)
        elif subset == 'test':
            self.seperate_data(0.2)
        else:
            pass

          
    def shuffle_data(self):

        # Keep corresponding title, code, descriptions and labels matched
        order = np.arange(0,len(self.title))
        np.random.shuffle(order)
        title_ = list()
        code_ = list()
        description_ = list()
        label_ = list()
        for i in order:
            title_.append(self.title[i])
            code_.append(self.code[i])
            description_.append(self.description[i])
            label_.append(self.label[i])

        self.title = title_
        self.code = code_
        self.description = description_
        self.label = label_

    def seperate_data(self,percent):

        # Seperate data with given percentage:
        title_ = list()
        code_ = list()
        description_ = list()
        label_ = list()
        for i in range(int(percent*len(self.title))):
            title_.append(self.title[i])
            code_.append(self.code[i])
            description_.append(self.description[i])
            label_.append(self.label[i])

        self.title = title_
        self.code = code_
        self.description = description_
        self.label = label_
