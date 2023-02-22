"""
This file implements a very basic tokenization function.
You may want to re-implement it or to use an external tokenization library, such as razdel.
"""
import re

TOKEN = re.compile(r'([^\W\d]+|\d+|[^\w\s])', re.U)


def re_tokenize(text):
    return TOKEN.findall(text)
