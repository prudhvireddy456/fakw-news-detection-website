from joblib import load
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import re
import string
# nltk.download('stopwords')
# nltk.download('punkt')

def preprocessing(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation) ,'',text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation) ,'',text)  #Removes all punctuation characters using string.punctuation
    words = word_tokenize(text)                 #Tokenize the words
    stop_words = set(stopwords.words('english'))    #stores all stopwords in english as a set stop_words
    words = [word for word in words if word not in stop_words]  #remove stopwords
    stemmer = PorterStemmer()               #porterstemmer initializes to stemmer 
    words = [stemmer.stem(word) for word in words]  #stemming process
    text = ' '.join(words)                  #Joins all the words after stopwords removal and stemmming as a text
    return text

def predict_RF(data):
    data=np.array(data).reshape(-1)
    model=load(r'E:\django\assignment\myworld\Scripts\fakenews\app1\RF.pk1')
    data=pd.Series(data)
    # data.columns=['text']
    data = data.apply(preprocessing)
    vectorization = load(r'E:\django\assignment\myworld\Scripts\fakenews\app1\f.joblib')
    data=vectorization.transform(data)
    result=model.predict(data)
    return result[0]
print(predict_RF("WASHINGTON (Reuters) - Trump campaign adviser George Papadopoulos told an Australian diplomat in May 2016 that Russia had political dirt on Democratic presidential candidate Hillary Clinton, the New York Times reported on Saturday. The conversation between Papadopoulos and the diplomat, Alexander Downer, in London was a driving factor behind the FBIâ€™s decision to open a counter-intelligence investigation of Moscowâ€™s contacts with the Trump campaign, the Times reported. Two months after the meeting, Australian officials passed the information that came from Papadopoulos to their American counterparts when leaked Democratic emails began appearing online, according to the newspaper, which cited four current and former U.S. and foreign officials. Besides the information from the Australians, the probe by the Federal Bureau of Investigation was also propelled by intelligence from other friendly governments, including the British and Dutch, the Times said. Papadopoulos, a Chicago-based international energy lawyer, pleaded guilty on Oct. 30 to lying to FBI agents about contacts with people who claimed to have ties to top Russian officials. It was the first criminal charge alleging links between the Trump campaign and Russia. The White House has played down the former aideâ€™s campaign role, saying it was â€œextremely limitedâ€ and that any actions he took would have been on his own. The New York Times, however, reported that Papadopoulos helped set up a meeting between then-candidate Donald Trump and Egyptian President Abdel Fattah al-Sisi and edited the outline of Trumpâ€™s first major foreign policy speech in April 2016. The federal investigation, which is now being led by Special Counsel Robert Mueller, has hung over Trumpâ€™s White House since he took office almost a year ago. Some Trump allies have recently accused Muellerâ€™s team of being biased against the Republican president. Lawyers for Papadopoulos did not immediately respond to requests by Reuters for comment. Muellerâ€™s office declined to comment. Trumpâ€™s White House attorney, Ty Cobb, declined to comment on the New York Times report. â€œOut of respect for the special counsel and his process, we are not commenting on matters such as this,â€ he said in a statement. Mueller has charged four Trump associates, including Papadopoulos, in his investigation. Russia has denied interfering in the U.S. election and Trump has said there was no collusion between his campaign and Moscow. "))
def predict_LG(data):
    data=np.array(data).reshape(-1)
    model=load(r'E:\django\assignment\myworld\Scripts\fakenews\app1\LR.pk1')
    data=pd.Series(data)
    # data.columns=['text']
    data = data.apply(preprocessing)
    vectorization = load(r'E:\django\assignment\myworld\Scripts\fakenews\app1\f.joblib')
    data=vectorization.transform(data)
    result=model.predict(data)
    return result[0]
# print(predict_LG("this is a fake news"))
