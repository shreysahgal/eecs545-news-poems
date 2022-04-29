import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from time import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, \
    Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import pickle

from utils import text_strip

original_data = pd.read_csv("data/NewsArticles.csv", encoding='iso-8859-1')
data = pd.DataFrame()
data["text"] = original_data["text"]
data["headline"] = original_data["title"]

processed_text = text_strip(data["text"])
processed_summary = text_strip(data["headline"])

nlp = spacy.load('en_core_web_lg', disable=['ner', 'parser']) 
text = [str(doc) for doc in nlp.pipe(processed_text, batch_size=5000)]
summary = ['_START_ '+ str(doc) + ' _END_' for doc in nlp.pipe(processed_summary, batch_size=5000)]

MAX_TEXT_LEN = 1000
MAX_SUMMARY_LEN = 25

# take text/summaries which fall below max lens
text = np.array(text)
summary= np.array(summary)

short_text = []
short_summary = []

for i in range(len(text)):
    if len(summary[i].split()) <= MAX_SUMMARY_LEN and len(text[i].split()) <= MAX_TEXT_LEN:
        short_text.append(text[i])
        short_summary.append(summary[i])
        
final_data = pd.DataFrame({'text': short_text,'summary': short_summary})
final_data['summary'] = final_data['summary'].apply(lambda x: 'sostok ' + x + ' eostok')

x_tr, x_val, y_tr, y_val = train_test_split(
    np.array(final_data["text"]),
    np.array(final_data["summary"]),
    test_size=0.1,
    random_state=0,
    shuffle=True,
)

# prepare input tokenizer
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_tr))

thresh = 2
cnt = 0
tot_cnt = 0

for key, value in x_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    if value < thresh:
        cnt += 1
    
# make input tokenizer
x_tokenizer = Tokenizer(num_words = tot_cnt - cnt) 
x_tokenizer.fit_on_texts(list(x_tr))
x_tr_seq = x_tokenizer.texts_to_sequences(x_tr) 
x_val_seq = x_tokenizer.texts_to_sequences(x_val)
x_tr = pad_sequences(x_tr_seq,  maxlen=MAX_TEXT_LEN, padding='post')
x_val = pad_sequences(x_val_seq, maxlen=MAX_TEXT_LEN, padding='post')
x_voc = x_tokenizer.num_words + 1  # Size of vocabulary (+1 for padding token)


# prepare output tokenizer
y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_tr))

thresh = 2
cnt = 0
tot_cnt = 0

for key, value in y_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    if value < thresh:
        cnt = cnt + 1

# make output tokenizer
y_tokenizer = Tokenizer(num_words=tot_cnt-cnt) 
y_tokenizer.fit_on_texts(list(y_tr))
y_tr_seq = y_tokenizer.texts_to_sequences(y_tr) 
y_val_seq = y_tokenizer.texts_to_sequences(y_val) 
y_tr = pad_sequences(y_tr_seq, maxlen=MAX_SUMMARY_LEN, padding='post')
y_val = pad_sequences(y_val_seq, maxlen=MAX_SUMMARY_LEN, padding='post')
y_voc = y_tokenizer.num_words + 1


latent_dim = 300
embedding_dim = 200

# define model
encoder_inputs = Input(shape=(MAX_TEXT_LEN, ))
enc_emb = Embedding(x_voc, embedding_dim, trainable=False)(encoder_inputs)
encoder_lstm1 = LSTM(latent_dim, return_sequences=True,
                     return_state=True, dropout=0.2,
                     recurrent_dropout=0.2)
(encoder_output1, state_h1, state_c1) = encoder_lstm1(enc_emb)
encoder_lstm2 = LSTM(latent_dim, return_sequences=True,
                     return_state=True, dropout=0.2,
                     recurrent_dropout=0.2)
(encoder_output2, state_h2, state_c2) = encoder_lstm2(encoder_output1)
encoder_lstm3 = LSTM(latent_dim, return_state=True,
                     return_sequences=True, dropout=0.2,
                     recurrent_dropout=0.2)
(encoder_outputs, state_h, state_c) = encoder_lstm3(encoder_output2)
decoder_inputs = Input(shape=(None, ))
dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=False)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True,
                    return_state=True, dropout=0.4,
                    recurrent_dropout=0.2)
(decoder_outputs, decoder_fwd_state, decoder_back_state) = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# create model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

# train
history = model.fit(
    [x_tr, y_tr[:, :-1]],
    y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:],
    epochs=50,
    callbacks=[es],
    batch_size=128,
    validation_data=([x_val, y_val[:, :-1]],
                     y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:
                     , 1:]),
    )

# plot loss
from matplotlib import pyplot

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('loss.png')

# create inference model
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs,
                      state_h, state_c])
decoder_state_input_h = Input(shape=(latent_dim, ))
decoder_state_input_c = Input(shape=(latent_dim, ))
decoder_hidden_state_input = Input(shape=(MAX_TEXT_LEN, latent_dim))
dec_emb2 = dec_emb_layer(decoder_inputs)
(decoder_outputs2, state_h2, state_c2) = decoder_lstm(dec_emb2,
        initial_state=[decoder_state_input_h, decoder_state_input_c])
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input,
                      decoder_state_input_h, decoder_state_input_c],
                      [decoder_outputs2] + [state_h2, state_c2])

# save everything
model.save('model')
encoder_model.save('encoder_model')
decoder_model.save('decoder_model')
with open('x_tokenizer.pickle', 'wb') as handle:
    pickle.dump(x_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('y_tokenizer.pickle', 'wb') as handle:
    pickle.dump(y_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)