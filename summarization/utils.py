import re
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_word = stop_words = set(stopwords.words('english'))

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
        row = ' '.join([i for i in row.split(' ') if i not in stop_words])

        yield row
