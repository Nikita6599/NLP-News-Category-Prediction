%matplotlib inline
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
from keras import utils as np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding,Input,LSTM
from keras.layers import Conv1D, GlobalMaxPooling1D
import tensorflow as tf
from sklearn.externals import joblib
from textblob import TextBlob
from keras.optimizers import RMSprop,Adam
from keras.callbacks import EarlyStopping



df = pd.read_excel("Data_Train.xlsx")
df.head()

df['SECTION'].value_counts()
df['SECTION'].isnull().sum()




#cleaning text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    text=re.sub('[^a-zA-Z]',' ',text)
    text=text.lower()
    text=text.split()
    lemmatizer = WordNetLemmatizer() 
    text=[lemmatizer.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text=' '.join(text)

    return text


df['STORY']=df['STORY'].map(lambda story:clean_text(story))
df['STORY'][0]

#Saving the cleaned data
cleaned_df=pd.DataFrame()
cleaned_df['STORY']=df['STORY']
cleaned_df['SECTION']=df['SECTION']
cleaned_df.to_csv('Cleaned_Train_data.csv')

#Load cleaned data
df=pd.read_csv('Cleaned_Train_data.csv')


#Removing common words
freq = pd.Series(' '.join(df['STORY']).split()).value_counts()[:10]
freq = list(freq.index)
df['STORY'] = df['STORY'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df['STORY'].head()

#Removing rare words
freq_rare = pd.Series(' '.join(df['STORY']).split()).value_counts()[-10:]
freq_rare = list(freq_rare.index)
df['STORY'] = df['STORY'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_rare))
df['STORY'].head()




#splitting the dataset
train, test = train_test_split(df, random_state=42, test_size=0.2)
x_train = train.STORY
x_test = test.STORY
y_train=train.SECTION
y_test=test.SECTION
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#TFIDF Vector
vectorizer=TfidfVectorizer(max_df=0.80,min_df=1,stop_words='english')
train_vectors=vectorizer.fit_transform(x_train)
test_vectors=vectorizer.transform(x_test)
total_vectors=vectorizer.transform(df['STORY'])

print(train_vectors.shape)
print(test_vectors.shape)


#Logistic(0.977064)
'''from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=31, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40,max_iter=50)
log_reg.fit(train_vectors, y_train)
log_reg_prediction = log_reg.predict(test_vectors)
accuracy_score(y_test,log_reg_prediction)
'''



#MLP other
'''from sklearn.neural_network import MLPClassifier
mlp_other=MLPClassifier(activation='relu',learning_rate='constant',alpha=0.01,learning_rate_init=0.01)
mlp_other.fit(train_vectors,y_train)
mlp_other_prediction=mlp_other.predict(test_vectors)
accuracy_score(y_test, mlp_other_prediction)
'''

#LSTM
'''
max_words = 1000
max_len = 8664
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x_train)
sequences = tok.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

y_train_cat=np_utils.to_categorical(y_train)
y_test_cat=np_utils.to_categorical(y_test)


def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(4,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

rnn_model=RNN()
rnn_model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
rnn_model.fit(sequences_matrix,y_train_cat,batch_size=32,epochs=10,
          validation_split=0.2)


test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

accr = rnn_model.evaluate(test_sequences_matrix,y_test_cat)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
'''


#MLP(0.979685)
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier()
mlp.fit(train_vectors,y_train)
mlp_prediction=mlp.predict(test_vectors)
accuracy_score(y_test, mlp_prediction)
joblib.dump(mlp,'mlp_model(0.98165137).pkl')

#training on all data
mlp_all=MLPClassifier()
mlp_all.fit(total_vectors,df['SECTION'])
joblib.dump(mlp_all,'mlp_model(0.98099606)_on_all.pkl')

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=mlp_all,X=total_vectors,y=df['SECTION'],cv=10)
accuracies.mean()


'''#LinearSVC(0.9777195)
from sklearn.svm import LinearSVC
linear_svc=LinearSVC(C=2.1)
linear_svc.fit(train_vectors,train['SECTION'])
linear_svc_prediction=linear_svc.predict(test_vectors)
accuracy_score(test['SECTION'], linear_svc_prediction)
joblib.dump(linear_svc,'linear_svc_model(0.9777195).pkl')
'''

#test for mlp
df_test=pd.read_excel('Data_Test.xlsx')
df_test['STORY']=df_test['STORY'].map(lambda story:clean_text(story))

#Removing common words
freq_test = pd.Series(' '.join(df_test['STORY']).split()).value_counts()[:10]
freq_test = list(freq_test.index)
df_test['STORY'] = df_test['STORY'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_test))
df_test['STORY'].head()

#Removing rare words
freq_rare_test = pd.Series(' '.join(df_test['STORY']).split()).value_counts()[-10:]
freq_rare_test = list(freq_rare_test.index)
df_test['STORY'] = df_test['STORY'].apply(lambda x: " ".join(x for x in x.split() if x not in freq_rare_test))
df_test['STORY'].head()

mlp_load=joblib.load('mlp_model(0.98165137).pkl')

testset_vectors=vectorizer.transform(df_test['STORY'])
test_mlp_all_prediction=mlp.predict(testset_vectors)
test_mlp_all_prediction=list(test_mlp_all_prediction)

dataframe_mlp_all=pd.DataFrame()
dataframe_mlp_all['STORY']=df_test['STORY']
dataframe_mlp_all['SECTION']=test_mlp_all_prediction

dataframe_mlp_all.to_excel('mlp_updated_all_submission.xlsx',index=False)

