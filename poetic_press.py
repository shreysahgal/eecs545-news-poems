from tensorflow import keras
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
import pandas as pd
import random
import itertools
import sys

from poetic_priors.utils import *

if __name__ == '__main__':
    # load model
    encoder_model = keras.models.load_model('summarization/model/encoder_model')
    decoder_model = keras.models.load_model('summarization/model/decoder_model')

    # load vocabularies
    with open('summarization/model/x_tokenizer.pickle', 'rb') as handle:
        x_tokenizer = pickle.load(handle)

    with open('summarization/model/y_tokenizer.pickle', 'rb') as handle:
        y_tokenizer = pickle.load(handle)
    
    # load word poeticness prior
    with open('poetic_priors/word_poeticness.npy', 'rb') as f:
        prior_poeticness = np.array(np.load(f))

    reverse_target_word_index = y_tokenizer.index_word
    reverse_source_word_index = x_tokenizer.index_word
    target_word_index = y_tokenizer.word_index

    # set maximum lengths
    max_text_len = 62
    max_summary_len = 15

    # load data
    data = pd.read_csv("summarization/data/NewsArticles.csv", encoding='iso-8859-1')

    # get 5 random samples
    idxs = random.sample(range(len(data["text"])), 5)
    original_texts = [data["text"][i] for i in idxs]
    original_summs = [data["title"][i] for i in idxs]

    texts = text_strip(original_texts)
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser']) 
    input_text = [str(doc) for doc in nlp.pipe(texts, batch_size=5000)]
    input_sequence = x_tokenizer.texts_to_sequences(input_text) 
    input_sequence = pad_sequences(input_sequence, maxlen=max_text_len, padding='post')

    # generate poem prior
    RHYME_TARGET = 'station'
    poem_prior = generate_prior(RHYME_TARGET, dict(itertools.islice(target_word_index.items(), 12856)))

    eval_data = pd.DataFrame()

    for i in range(len(original_texts)):
        sequence = input_sequence[i]

        if original_summs:
            print('original headline: ', original_summs[i])

        summ = decode_sequence(
            sequence.reshape(1, max_text_len), 
            encoder_model, 
            decoder_model, 
            target_word_index, 
            reverse_target_word_index, 
            max_summary_len,
            poem_prior,
            prior_poeticness
        )

        print('Predicted summary: ', summ)
        print()

        eval_data = eval_data.append({'input_text': original_texts[i], 'output_poem': summ}, ignore_index=True)

    eval_data.to_csv("eval_data.csv")
