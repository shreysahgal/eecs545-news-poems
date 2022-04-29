import re
import numpy as np
import pronouncing
import torch
import itertools
from .rhymer import get_phones, get_slant_rhymes
from scipy import stats

def softmax(x):
    return np.array(np.exp(x)/np.sum(np.exp(x))).reshape(-1)


def text_strip(column):

    for row in column:
        row = re.sub("(\\t)", " ", str(row)).lower()
        row = re.sub("(\\r)", " ", str(row)).lower()
        row = re.sub("(\\n)", " ", str(row)).lower()
        row = re.sub("(__+)", " ", str(row)).lower()
        row = re.sub("(--+)", " ", str(row)).lower()
        row = re.sub("(~~+)", " ", str(row)).lower()
        row = re.sub("(\+\++)", " ", str(row)).lower()
        row = re.sub("(\.\.+)", " ", str(row)).lower()
        row = re.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", " ", str(row)).lower()
        row = re.sub("(mailto:)", " ", str(row)).lower()
        row = re.sub(r"(\\x9\d)", " ", str(row)).lower()
        row = re.sub("(\.\s+)", " ", str(row)).lower()
        row = re.sub("(\-\s+)", " ", str(row)).lower()
        row = re.sub("(\:\s+)", " ", str(row)).lower()

        try:
            url = re.search(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", str(row))
            repl_url = url.group(3)
            row = re.sub(r"((https*:\/*)([^\/\s]+))(.[^\s]+)", repl_url, str(row))
        except:
            pass

        row = re.sub("(\s+)", " ", str(row)).lower()
        row = re.sub("(\s+.\s+)", " ", str(row)).lower()

        yield row


def seq2summary(input_seq, target_word_index, reverse_target_word_index):
    newString = ''
    for i in input_seq:
        if i != 0 and i != target_word_index['sostok'] and i != target_word_index['eostok']:
            newString = newString + reverse_target_word_index[i] + ' '

    return newString


def seq2text(input_seq, reverse_source_word_index):
    newString = ''
    for i in input_seq:
        if i != 0:
            newString = newString + reverse_source_word_index[i] + ' '

    return newString


def generate_prior(rhyme_word, vocabulary):

    rhyme_scale_factor = 500
    slant_scale_factor = 300

    prior_vector = np.ones((len(vocabulary), 1))

    rhymes = pronouncing.rhymes(rhyme_word)
    slant_rhymes = get_slant_rhymes(get_phones(rhyme_word))
    
    for r in rhymes:
        if r in vocabulary.keys():
            prior_vector[vocabulary[r]] = rhyme_scale_factor
    
    for r in slant_rhymes:
        if r in vocabulary.keys():
            prior_vector[vocabulary[r]] = slant_scale_factor

    return prior_vector


def update_syllable_count(sequence):
    last_word = sequence.split()[-1]
    phones = pronouncing.phones_for_word(last_word)
    try:
        count = pronouncing.syllable_count(phones[0])
    except:
        count = 2
    return count


def decode_sequence(input_seq, encoder_model, decoder_model, target_word_index,
                        reverse_target_word_index, max_summary_len, prior, prior_poeticness):

    (e_out, e_h, e_c) = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    syllable_sum = 0

    while not stop_condition:
        (output_tokens, h, c) = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        if syllable_sum >= 6:
            tmp = np.multiply(output_tokens[0, -1, :].reshape(-1, 1), prior)
            tmp = tmp.reshape(-1)
            tmp = np.multiply(prior_poeticness, tmp)
            sampled_token_index = np.argmax(tmp)
            syllable_sum = 0 # clear running sum

        else:
            tmp = softmax(output_tokens[0, -1, :])
            sampled_token_index = np.multiply(prior_poeticness,tmp)
            sampled_token_index = np.argmax(sampled_token_index)

        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'eostok':
            decoded_sentence += ' ' + sampled_token

        if sampled_token == 'eostok' or len(decoded_sentence.split()) >= max_summary_len - 1:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        syllable_sum += update_syllable_count(decoded_sentence)      

        (e_h, e_c) = (h, c)

    return decoded_sentence
