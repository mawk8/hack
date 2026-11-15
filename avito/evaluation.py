"""Classification metrics independent of training and model dependencies."""
from collections import Counter


def metrics(expected, predicted):
    counts = Counter()
    for truth, result in zip(expected, predicted):
        truth, result = set(truth), set(result)
        counts.update(tp=len(truth & result), fp=len(result - truth), fn=len(truth - result))
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    return dict(tp=tp, fp=fp, fn=fn, micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
