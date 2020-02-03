from nltk.tokenize import word_tokenize
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import math 
import pandas as pd
import collections

stemmer= PorterStemmer()


txt = dict()
result2 = list()
data = pd.read_csv('reviews.csv')  
df = pd.DataFrame(data, columns = ['text', 'categories']) 
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer() 


result = list()
result1 = set()
result2 = list()
k = 0
for x in df['text']:
    result.append(df['text'][k].split(" "))
    k = k + 1
k = 0

for x in df['categories']:
    if(k<100000):
        result2.append(df['categories'][k].split(";"))


n = 0
for i in result:
    for word in i:
        if n in txt:
            txt[n] = str(txt[n]) + " " + word
        else:
            txt[n] = word
    n = n + 1

txt2 = dict()
txt3 = dict()
txt4 = dict()
count = dict()
count_doc = dict()
countall = dict()
tf = dict()
idf = dict()
all_word_one = dict()
tfidf = dict()
tfidf_final = dict()
final = dict()
fi = list()

for key, value in txt.items():
    txt2[key] = value.translate(str.maketrans('', '', string.punctuation)) 

for key, value in txt2.items():
    txt2[key] = word_tokenize(value)
i = 0
for key, value in txt2.items():
    for word in value:
        if word not in stop_words and word  != "I" and word !="this" and word != 'This':
            if key not in txt3:
                txt3[key] =  word
            else:
                txt3[key] = txt3[key] +  " " + word

for key,value in txt3.items():
    value2 = value.split()
    for word in value2:
        if word not in count:
            count[word] =  1
        else:
            count[word] = count[word] + 1
    value2 = ()
    countall[key] = count
    count = dict()
k = 0

for key,value in txt3.items():
    value2 = value.split()
    for word in value2:
        if word not in count_doc:
            count_doc[word] = 1;
        else:
            count_doc[word] = count_doc[word] + 1

for key, value in countall.items():
    for k, v in value.items():
       tf[k] = v/len(value)
    all_word_one[key] = tf
    tf = dict()

for key, value in count_doc.items():
    idf[key] = math.log(len(count_doc)/value)

for key, value in all_word_one.items():
    for k, v in value.items():
        if k in idf:
            tfidf[k] = v * idf[k]
    tfidf = collections.OrderedDict(tfidf)

    tfidf_final[key] = tfidf
    
    tfidf = dict()
 
for key, value in tfidf_final.items():
    j = 0
    for k, v in value.items():
        if j < 3:
            fi.append(k)
            final[key] = fi 
            j = j + 1
        else:
            fi = []
            j = j + 1

correct = 0
incorrect = 0
def common_data(list1, list2): 

  
    # traverse in the 1st list 
    for x in list1: 
  
        # traverse in the 2nd list 
        for y in list2: 
    
            # if one common 
            if x == y: 
                return True  
            else:
                return False

for z in range(100000):
    if common_data(final[z],result[z]):
        correct = correct + 1
    else:
        incorrect = incorrect + 1
