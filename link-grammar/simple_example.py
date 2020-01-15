from linkgrammar import *
# user_input = raw_input("Input an example sentence: ")
user_input = "This is a simple sentence."
user_input = "We will also compare and contrast a traditional diet with the modern Westernised UK diet and evaluate the respective abilities to maintain health and environment."
sent = Sentence(user_input, Dictionary(), ParseOptions())
linkages = sent.parse()
print Clinkgrammar.sentence_num_linkages_found(sent._obj)
for linkage in linkages:
    print linkage.diagram()
    print (type(linkage.postscript()))
    links = linkage.postscript().split("]")
    for link in links:
        print link
    break
