import linkgrammar
import re
import json
from nltk import tokenize

# Parse Options:
po = linkgrammar.ParseOptions(min_null_count=0, max_null_count=999)

# Dictionary:
en_dir = linkgrammar.Dictionary()

count = 0

# Abbreviations:
regex1 = re.compile('(i\.e\.)')
regex2 = re.compile('(e\.g\.|e\.g)')
regex3 = re.compile('(etc|etc.)')


def s(q):
    return '' if q == 1 else 's'


def linkage_stat(psent, lang, lkgs, sent_po):
    """
    This function mimics the linkage status report style of link-parser
    """
    random = ' of {} random linkages'. \
             format(linkgrammar.Clinkgrammar.sentence_num_linkages_post_processed((psent._obj))) \
             if linkgrammar.Clinkgrammar.sentence_num_linkages_found(psent._obj) > sent_po.linkage_limit else ''

    # print ('{}: Found {} linkage{} ({}{} had no P.P. violations)'. \
    #       format(lang, linkgrammar.Clinkgrammar.sentence_num_linkages_found(psent._obj),
    #              s(linkgrammar.Clinkgrammar.sentence_num_linkages_found(psent._obj)), len(lkgs), random))



with open("../test.json", 'r') as file:
    data = json.load(file)
    for element in data['table']:
        course_description = element['description']
        print(course_description)
        # course_description = "It will provide opportunity for students to develop strategies for enhancing and improving the performance of multi-professional initiatives."
        description = regex3.sub('', regex2.sub('for example', regex1.sub('in other words', course_description)))
        sentence_list = tokenize.sent_tokenize(description)
        for sentence in sentence_list:
            sent = linkgrammar.Sentence(str(sentence), en_dir, po)
            linkages = sent.parse()
            linkage_stat(sent, 'English', linkages, po)

            if len(linkages) > 0:
                count += 1

            for linkage in linkages:
                links = linkage.postscript().split("]")
                for link in links:
                    print(link)
                break
        print(count)



# description = ["This module aims to support students as they seek to make sense of the increasingly complex practice environment."]
# sent = linkgrammar.Sentence(description, linkgrammar.Dictionary(), linkgrammar.ParseOptions())
# linkages = sent.parse()
