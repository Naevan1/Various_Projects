#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install tensorflow')


# In[1]:


import pandas as pd 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint


# In[2]:


pd.set_option('display.max_colwidth', None)


# In[3]:


df = pd.read_csv('/home/jovyan/LSTM/data/SINGAPORE FULL PATHS FEB 23-MAY 23_csv.csv')
df.drop(columns='Unnamed: 0',inplace=True)
df.dropna(inplace=True)
df = df[df['Site Path'].str.contains('en-sg')]
df.reset_index(inplace=True)
df.drop(columns='index',inplace=True)
df.rename(columns={'Unnamed: 3':'Percent'},inplace=True)


# In[4]:


df['Site Path'][0]


# In[5]:


df['Site Path'] = df['Site Path'].str.replace('EXP\$\|\$', '', regex=True)


# In[6]:


df['Site Path'] = df['Site Path'].str.replace('EXP\$\|', '', regex=True)


# In[7]:


df['Site Path'][0]


# I am thinking of creating 2 LSTM , one with Entered&Exit and one without

# In[8]:


df['count'] = df['Site Path'].str.count('>')


# In[9]:


df['count'].value_counts()


# In[10]:


df['Site Path'] = df['Site Path'].str.replace('Entered Site > ', '', regex=True)


# In[11]:


# Add label 1 to sequences that contain 'Thank' keyword

df['label'] = df['Site Path'].str.contains('thank').astype(int)


# In[12]:


df['label'].value_counts()
# We have 458 negataive and 42 positive, so roughly 9% positive labels. 


# In[13]:


df['Site Path'][41]


# In[14]:


df['Site Path'][43]


# ## data

# In[62]:


df


# # modeling 

# In[137]:


data = list(df['Site Path'])  # your data here
labels = list(df['label'])  # your labels here


# In[138]:


tokenizer = Tokenizer(filters='', split=' > ',oov_token='OOV')


# In[139]:


tokenizer.fit_on_texts(data)


# In[140]:


# Vocabulary
tokenizer.word_index
#OOV words will be ignored / discarded by default, if oov_token is None, LATER when we predict.:


# In[141]:


sequences = tokenizer.texts_to_sequences(data)


# In[142]:


sequences


# In[143]:


# The tokens which are thank , which we should remove from the sequences. The reason we remove this is because the model is very biased if itt contains this.
# for example,whenever it sees a thank you page it will immediately think its positive, and negative otherwise. But it will not be learning if its like this.
# So we have to remove them, and also remove any integers/urls after those pages too.
thank_tokens = [value for key, value in tokenizer.word_index.items() if 'thank' in key]


# In[144]:


def cut_after_thank(sequence, thank_tokens):
    for i in range(len(sequence)):
        if sequence[i] in thank_tokens:
            return sequence[:i+1]
    return sequence

sequences = [cut_after_thank(sequence, thank_tokens) for sequence in sequences]


# In[145]:


# Remove the thank you token
sequences = [[token for token in sequence if token not in thank_tokens] for sequence in sequences]


# In[146]:


num_unique_urls = len(tokenizer.word_index) + 1  # Add one for padding


# In[147]:


max_sequence_len = max([len(seq) for seq in sequences])


# In[148]:


sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')


# In[149]:


sequences[0]


# In[150]:


# Remove the last URL 
padded_sequences = np.array([seq[:-1] for seq in sequences])


# In[151]:


labels = np.array(labels)


# In[152]:


print(sequences.shape)
print(padded_sequences.shape)  # Shape of your input data
print(labels.shape)  # Shape of your labels


# In[153]:


# To decode, you put the numpy array into a list and you see what it is.
tokenizer.sequences_to_texts([sequences[0]])[0]


# ## Creating a Binary Classification LSTM model 
# ## `Frame the problem as a binary classification task, where the goal is to predict whether a sequence leads to a "lead" page or not.`

# In[154]:


# Creating the binary classification model
clf = Sequential()
clf.add(Embedding(input_dim=num_unique_urls, output_dim=10, input_length=max_sequence_len - 1))
clf.add(LSTM(32))
clf.add(Dense(1, activation='sigmoid'))  # Binary classification
clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[155]:


clf.summary()


# In[156]:


from sklearn.model_selection import train_test_split

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.4, random_state=42, stratify=labels)

# further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)  # 0.25 x 0.8 = 0.2


# In[157]:


X_train[:5]


# In[158]:


from keras.callbacks import EarlyStopping
# define early stopping callback
earlystop = EarlyStopping(monitor='val_accuracy', patience=20)


# In[159]:


# Train
clf.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_val,y_val),batch_size=8,callbacks=[earlystop])


# In[160]:


loss, accuracy = clf.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)


# In[161]:


y_pred = clf.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(12,8))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# 3. Calculate the AUC
auc = roc_auc_score(y_test, y_pred)
print('AUC: %.3f' % auc)


# In[162]:


from sklearn.metrics import roc_curve, auc
y_pred = clf.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plotting
plt.figure(figsize=(10,6))
lw = 2  # Line width
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# ## Applying the binary clf i.e. given a sequence predict if next URL will be a thank you or not

# In[ ]:


# def predict_next_url(model, tokenizer, sequence):
#     token_list = tokenizer.texts_to_sequences([sequence])[0]
#     token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
#     predicted_url_index = model.predict_classes(token_list, verbose=0)[0]
#     for url, index in tokenizer.word_index.items():
#         if index == predicted_url_index:
#             return url


# In[36]:


def predict_lead_page(model, tokenizer, sequence):
    token_list = tokenizer.texts_to_sequences([sequence])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len -1, padding='pre')
    prediction = model.predict(token_list, verbose=0)
    return prediction


# In[80]:


df.iloc[1]


# In[85]:


# define a new sequence for which you want to predict if it leads to a 'thank' page
new_sequence = '/discover/en-sg/offer-services$|$Super fast delivery across the globe from only $9* > /discover/en-sg$|$DHL Express Singapore - E-commerce, business, logistics advice'

# use the function to predict if the new_sequence leads to a 'thank' page
prediction = predict_lead_page(model, tokenizer, new_sequence)

# print out the prediction
print(f"Probability of lead page: {prediction}")
print(f"Probability of lead page: {prediction[0]}")
print(f"Probability of lead page: {prediction[0][0]}")

if prediction > 0.5:
    print("This sequence is likely to lead to a 'thank' page.")
else:
    print("This sequence is not likely to lead to a 'thank' page.")


# Since you've trained your model on a dataset where 'thank' pages are relatively rare, it might have learned to associate high probability with sequences that are somehow 'unusual' or 'abnormal', because the 'thank' sequences were unusual in the context of the overall training set.

# In[82]:


sequences[1]


# ### To propose pages that could lead to 'thank' pages, you would need to apply a slightly different strategy. One approach is to use a trained model that can predict the next URL, and given a starting sequence, generate sequences by iteratively choosing URLs that are likely to result in a 'thank' page.

# In[343]:


def propose_sequence(model, tokenizer, start_sequence, num_steps):
    #give the sequence you want to predict on
    sequence = start_sequence
    for _ in range(num_steps):
        token_list = tokenizer.texts_to_sequences([sequence])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        # Feed the paddded sequence to model, generate a probability distribution over the 2 possible outcome
        probabilities = model.predict(token_list, verbose=0)[0]
        probabilities_sorted_indexes = np.argsort(probabilities)[::-1]
        for index in probabilities_sorted_indexes:
            proposed_url = ""
            for url, url_index in tokenizer.word_index.items():
                if url_index == index:
                    proposed_url = url
                    break
            if 'thank' in proposed_url:
                sequence += ' > ' + proposed_url
                break
    return sequence


# In[347]:


start_sequence = "/discover/en-sg/ship-with-dhl/start-shipping/6-simple-steps-to-start-shipping$|$6 Simple Steps > /discover/en-sg$|$DHL Express Singapore - E-commerce, business, logistics advice"
num_steps = 5
proposed_sequence = propose_sequence(model, tokenizer, start_sequence, num_steps)
print(proposed_sequence)


# In[352]:


start_sequence="discover/en-sg/offer-services$|$Super fast delivery across the globe from only $9* > /discover/en-sg/offer-services/thank-you$|$Thank you > /discover/en-sg$|$DHL Express Singapore - E-commerce, business, logistics advice"
proposed_sequence = propose_sequence(model, tokenizer, start_sequence, num_steps)
print(proposed_sequence)


# # URL Predictor Model

# In[163]:


from keras.utils import to_categorical


# In[164]:


data = list(df['Site Path'])  # your data here
labels = list(df['label'])  # your labels here
tokenizer = Tokenizer(filters='', split=' > ',oov_token='OOV')
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
num_unique_urls = len(tokenizer.word_index) + 1  # Add one for padding
max_sequence_len = max([len(seq) for seq in sequences])


# In[165]:


num_unique_urls,max_sequence_len


# In[166]:


X = []
y = []

for sequence in sequences:
    for i in range(1, len(sequence)):
        X.append(sequence[:i])
        y.append(sequence[i])
        
# Padding the sequences
X = pad_sequences(X, maxlen=max_sequence_len-1, padding='pre')

# one-hot encoding the labels
y = to_categorical(y, num_classes=num_unique_urls)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# defining the model
model = Sequential()
model.add(Embedding(input_dim=num_unique_urls, output_dim=50, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(num_unique_urls, activation='softmax')) # Multiclass classification
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[167]:


# An example of the one hot encoding . It is needed for softmax. 297 elements for a probability for every possible sequence.
y[0],y[0].shape


# In[168]:


X_train[0]


# In[169]:


hist = model.fit(X_train, y_train, epochs=200, verbose=1, validation_data=(X_val, y_val), batch_size=32)


# In[170]:


# summarize history for accuracy
plt.figure(figsize=(12, 6))
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.figure(figsize=(12, 6))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[171]:


def predict_next_url(model, tokenizer, sequence):
    token_list = tokenizer.texts_to_sequences([sequence])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_url_index = model.predict_classes(token_list, verbose=0)[0]
    for url, index in tokenizer.word_index.items():
        if index == predicted_url_index:
            return url


# In[172]:


y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)


# In[173]:


y_pred_class


# In[174]:


tokenizer.sequences_to_texts([[3]])


# In[175]:


def predict_next_url(model, tokenizer, sequence):
    """
    To use the function predict_next_url, you need to provide a sequence of URLs. 
    This sequence should be in the same format as the sequences you used to train your model. Here is an example: X > A > Exited Site
    """
    token_list = tokenizer.texts_to_sequences([sequence])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len -1, padding='pre')
    prediction = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(prediction, axis=-1)
    predicted_url = ""
    for url, url_index in tokenizer.word_index.items():
        if url_index == predicted_index:
            predicted_url = url
            break
    return predicted_url

import heapq

def predict_next_n_urls(model, tokenizer, sequence, n):
    """
    To use the function predict_next_url, you need to provide a sequence of URLs. 
    This sequence should be in the same format as the sequences you used to train your model. Here is an example: X > A > Exited Site
    """
    token_list = tokenizer.texts_to_sequences([sequence])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len -1, padding='pre')
    prediction = model.predict(token_list, verbose=0)[0]
    top_n_indices = heapq.nlargest(n, range(len(prediction)), prediction.take)
    top_n_urls = []
    for url, url_index in tokenizer.word_index.items():
        if url_index in top_n_indices:
            top_n_urls.append((url, prediction[url_index]))
    return top_n_urls


# In[176]:


predict_next_url(model,tokenizer,'/discover/en-sg/offer-services$|$Super fast delivery across the globe from only $9*')


# In[177]:


predict_next_n_urls(model,tokenizer,'/discover/en-sg/offer-services$|$Super fast delivery across the globe from only $9*',3)


# # Combining the 2 models. Method : 
# - Output the top 3 URL from the sequence URL prediction model
# - Pass those sequences to the Binary Classification model (predict if next seq will be thank you or not)
# - Return results

# In[194]:


def predict_next_url_combined(predictor_model, classifier_model, tokenizer, sequence, n_results=5):
    """
    This function uses the next-URL prediction model to create a probability distribution of all the possible next URL. It ranks them on probability.
    Then, for each of the next-possible-URL , we concatenate it into the base sequence which we gave as input and pass it through the classification model, which outputs probability of the next page being a thank you.
    We multiply the probability of the binary clssification with the the next-url probability(for now with no weights on each). We rank those on highest, and print the n_results many we want.
    The probability that we see in the end so is the multiplication of the 2 previous probabilities described, next url in general and also thank you page probability.
    """
    # Extract the URL from the sequence tuple
    sequence = sequence[0]

    # Get the next URL probabilities from the sequence predictor
    token_list = tokenizer.texts_to_sequences([sequence])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len -1, padding='pre')
    predictor_probs = predictor_model.predict(token_list, verbose=0)[0]

    # Get the classification probabilities for each URL
    classifier_probs = np.zeros_like(predictor_probs)
    for url, url_index in tokenizer.word_index.items():
        extended_sequence = sequence + ' > ' + url
        token_list = tokenizer.texts_to_sequences([extended_sequence])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        classifier_probs[url_index-1] = classifier_model.predict(token_list, verbose=0)[0]

    # Combine the probabilities by a simple multiplication, we can also add weight here
    combined_probs = predictor_probs * classifier_probs

    # Return the top n URLS with the highest combined probability
    top_n_indexes = np.argsort(combined_probs)[-n_results:][::-1]

    predicted_urls_probs = []
    for predicted_index in top_n_indexes:
        predicted_url = ""
        for url, url_index in tokenizer.word_index.items():
            if url_index - 1 == predicted_index:
                predicted_url = url
                break
        predicted_urls_probs.append((sequence + ' > ' + predicted_url, combined_probs[predicted_index]))

    return predicted_urls_probs


# In[195]:


predict_next_url_combined(model,clf,tokenizer,'/discover/en-sg/offer-services$|$Super fast delivery across the globe from only $9*')


# In[199]:


sequences = predict_next_n_urls(model, tokenizer, '/discover/en-sg/offer-services$|$Super fast delivery across the globe from only $9*', 3)
print('For our given sequence, these are the 3 most probable sequences. For each of the 3, the model provides 5 most promising URL that lead to a thank you page along with the confidence\n')
for sequence in sequences:
    pprint(predict_next_url_combined(model, clf, tokenizer, sequence))
    print('\n')

