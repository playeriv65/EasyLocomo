import regex
import string
import unicodedata
import numpy as np
from collections import Counter
from nltk.stem import PorterStemmer

try:
    from bert_score import score as bert_score_func
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("Warning: bert_score not installed. BERTScore metric will be unavailable.")

ps = PorterStemmer()

def normalize_answer(s):
    s = s.replace(',', "")
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score_single(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score_multi(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    return np.mean([max([f1_score_single(prediction, gt) for prediction in predictions]) for gt in ground_truths])

def calculate_bert_score(predictions, references, lang="en", verbose=False):
    if not BERT_SCORE_AVAILABLE:
        return [0.0] * len(predictions)
    
    # bert_score.score returns (P, R, F1)
    # We are interested in F1
    P, R, F1 = bert_score_func(predictions, references, lang=lang, verbose=verbose, rescale_with_baseline=True)
    return F1.tolist()
