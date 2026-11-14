"""Fixed classification baseline: only source101 is learnable from this train."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from avito.data import CATEGORIES, validate_description, validate_labels

THRESHOLD = 0.5
SEED = 42


def train(descriptions, label_sets):
    if (not isinstance(descriptions, (list, tuple)) or not descriptions
            or not isinstance(label_sets, (list, tuple)) or len(descriptions) != len(label_sets)):
        raise ValueError('Nonempty matching descriptions and label_sets required')
    for description in descriptions:
        validate_description(description)
    labels = [validate_labels(values) for values in label_sets]
    y = MultiLabelBinarizer(classes=CATEGORIES).fit_transform(labels)
    model = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                                max_features=80000, sublinear_tf=True)),
        ('classifier', OneVsRestClassifier(LogisticRegression(
            C=1.0, solver='liblinear', random_state=SEED, max_iter=1000))),
    ])
    return model.fit(descriptions, y)


def predict(model, description, source_mcId=101):
    if type(source_mcId) is not int or source_mcId not in range(101, 107):
        raise ValueError('source_mcId must be an integer 101..106')
    if source_mcId != 101:
        raise NotImplementedError('Baseline supports only source_mcId=101; category101 labels are unknown')
    validate_description(description)
    probabilities = np.asarray(model.predict_proba([description]))
    if (probabilities.shape != (1, len(CATEGORIES)) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)):
        raise ValueError('Model returned invalid category probabilities')
    return [category for category, probability in zip(CATEGORIES, probabilities[0]) if probability >= THRESHOLD]


def save_model(model, path):
    """Save a local pickle-based artifact without replacing an existing file."""
    with Path(path).open('xb') as stream:
        joblib.dump(model, stream)


def load_model(path, *, trusted=False):
    """Load only explicitly trusted local files: joblib can execute pickle code."""
    if trusted is not True:
        raise ValueError('Only trusted local models may be loaded; pass trusted=True')
    return joblib.load(Path(path))
