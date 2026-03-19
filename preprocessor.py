"""
preprocessor.py
---------------
Shared module imported by BOTH train_model.py and main.py.
Pickle needs the class to exist at the same import path when
saving AND loading — this file guarantees that.
"""

import re
from sklearn.base import BaseEstimator, TransformerMixin


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Clean raw text before TF-IDF vectorisation:
      - Strip HTML tags  (<br />, <b>, etc.)
      - Lowercase
      - Remove URLs
      - Remove non-alphanumeric characters (keep useful punctuation)
      - Collapse whitespace
    """
    _TAG_RE  = re.compile(r"<[^>]+>")
    _URL_RE  = re.compile(r"http\S+|www\S+")
    _JUNK_RE = re.compile(r"[^a-z0-9\s!?'.,]")
    _WS_RE   = re.compile(r"\s+")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for text in X:
            text = self._TAG_RE.sub(" ", text)
            text = text.lower()
            text = self._URL_RE.sub(" ", text)
            text = self._JUNK_RE.sub(" ", text)
            text = self._WS_RE.sub(" ", text).strip()
            out.append(text)
        return out
