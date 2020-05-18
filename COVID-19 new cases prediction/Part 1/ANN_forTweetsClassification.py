# Tweets Classification
# Label into 2 groups: Others tweets vs Covid tweets
# Step 1: Word Embedding 
# Step 2: Regression
    # Version 2.1: Logistic Regression by sklearn 
    # Version 2.2: ANN
# Step 3: Predict test set and Evaluate the model
# Step 4: Predict the real tweets that havn't labeled yet

#---------------------------------------------------------------------------------

# Step 1: Word Embedding 
# Ref: https://stackabuse.com/python-for-nlp-word-embeddings-for-deep-learning-in-keras/

from numpy import asarray
from numpy import zeros
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from keras.models import Sequential   
from keras.layers import Dense    
from keras.layers import Dropout   


import pandas as pd
import numpy as np
import re
import seaborn as sns

# Training set: Tweets with 2 labeled, Others vs Covid
data_train = pd.read_csv("E:\DataScience\DSTI\Artificial Neural Networks\Project\Final\Tweets_word_for_Train.csv")

# Plot the number of 0, Others tweets and 1, Covid tweets
data_train["Label"].replace({0:"Others tweets", 1:"Covid19 tweets"},inplace = True)
sns.countplot(x="Label", data = data_train)

# Data preprocessing
X = []
pstem = PorterStemmer()
for i in range(data_train['Text'].shape[0]):
    #Remove unwanted words
    tweet = re.sub("[^a-zA-Z]", ' ', data_train['Text'][i])
    #Transform words to lowercase
    tweet = tweet.lower()
    tweet = tweet.split()
    #Remove stopwords then Stemming it
    tweet = [pstem.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    #Append cleaned tweet to corpus
    X.append(tweet)


data_train["Label"].replace({"Others tweets":0, "Covid19 tweets":1},inplace = True)   
y = data_train["Label"].values


# Divide data into testing and training sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Textual data has to be converted into some sort of numeric form before it can be used by statisitical algorithms like machine and deep learning models
# This step is called "Word Embedding"
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test  = tokenizer.texts_to_sequences(X_test)

# Sentences can have different lengths
# and therefore the sequences returned by the Tokenizer class also consist of variable lengths
# We specify that maximum length of the sequence will be 2 (Because we work with Bi Term)
# For the sentences having length less than 2, the remaining indexes will be padded with zeros
# For the sentences having length greater than 2, the remaining indexes will be truncated
vocab_size = len(tokenizer.word_index) + 1

maxlen = 2

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test  = pad_sequences(X_test, padding='post',  maxlen=maxlen)

# Next, we need to load the built-in GloVe word embeddings
# Downloaded from https://nlp.stanford.edu/projects/glove/
embeddings_dictionary = dict()
glove_file = open('E:\DataScience\DSTI\Artificial Neural Networks\Project\Draft\glove.6B.50d.txt', encoding="utf8")


for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions

glove_file.close()

embedding_matrix = zeros((vocab_size, 50))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


#---------------------------------------------------------------------------------

# Step 2: Regression
# Version 2.2: ANN 

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))

# Adding the second hidden layer
# Add Dropout to avoid overfitting
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the third hidden layer
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the fourth hidden layer
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# We change activation function to sigmoid because we want the output as probability

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Stochastic Gradient Descent
# optimizer = algorithm to find the best weight 
# (so far the weight has just initialized but maybe not the best one)
# adam is one of stochastic gradient algorithm
# loss = loss function of Stochastic Gradient Descent (Easiest example = sum of square dif in linear regression)
# Here we have 2 output >> The best loss function = binary_crossentropy
# metrics = Criteria that you want to use to verify + improve the quality of your model 
# We choose accuracy so the model will keep improve until it reaches the maximum accuracy possible
# metrics need a list >> That is why we have to put [ ] cover 'accuracy'


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
# batch_size = No. of observation that you want to update the weight
# epochs = no. of rounds to train the model
# Note that result get the best accuracy = 0.8346 = 83% accuracy
# So we can expect this level of accuracy in test set 


#---------------------------------------------------------------------------------

# Step 3: Predict and Evaluate the model


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# .predict returns the probabilities
# Next we want to see the accuracy of our model by confusion matrix
# But confusion metric need input in form "True or False"
# So, we have to edit y_pred >> Set threshold 
# Here if .predict give less than 0.5 prob >> Return false
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Compute accuracy in console
# Accuracy = No. of correct predictions/Total no. of prediction


# Visualising the Training set results
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# meshgrid set the bound of pixels that we want to include in the frame 
# Take the minimum point - 1 and maximum + 1 because we don't want the points squeeze too close to the axis
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('ANN Classifier (Training set)', weight="heavy")
plt.legend()
plt.show()

#---------------------------------------------------------------------------------


# Step 4: Predict the real tweets that havn't labeled yet

tweets_forlabeled = pd.read_csv("E:\DataScience\DSTI\Artificial Neural Networks\Project\Final\Tweets_word_for_Test.csv")
tweets_predict = classifier.predict(tweets_forlabeled)
tweets_predict = (tweets_predict> 0.5)

# Merge labeled prediction to make final twitter file
tweets_label = pd.DataFrame(tweets_predict)
tweets_label_int = tweets_label.astype(int)
tweets_forTest_with_label = pd.concat([tweets_forlabeled, tweets_label_int], axis = 1)
tweets_forTest_with_label.to_csv('E:\DataScience\DSTI\Artificial Neural Networks\Project\Final\Tweets_labeled_ANN.csv', index=False)






















