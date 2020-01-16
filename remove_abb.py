import json
import re


regex1 = re.compile('(i\.e\.)')
regex2 = re.compile('(e\.g|e\.g\.)')
regex3 = re.compile('(etc|etc.)')
with open("test.json", 'r') as file:
    data = json.load(file)

    for element in data['table']:
        description = element['description']
        description = regex3.sub('', regex2.sub('for example', regex1.sub('in other words', description)))
        print(description)
