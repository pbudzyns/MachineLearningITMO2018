import numpy as np
from FeatureSelection.metrics import chi2, lda, anova, pearson


class SelectKFeatures:

    def __init__(self, score_func, k):
        self.score_func = score_func
        self.k = k

    def fit(self, features, labels):
        pass

    def transform(self, features):
        pass


def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append([int(x) for x in f.readline().split()])
    return np.array(data)


if __name__ == '__main__':

    arcene_train_data = load_data('arcene/arcene_train.data')
    arcene_train_labels = load_data('arcene/arcene_train.labels')

    arcene_valid_data = load_data('arcene/arcene_valid.data')
    arcene_valid_labels = load_data('arcene/arcene_valid.labels')

    metrics = [chi2, lda, anova, pearson]
    # For metric in metrics
    # fit data reducer with metric, learn model with reduced amount of features

    # validate model and check score