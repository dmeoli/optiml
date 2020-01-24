import copy
from collections import defaultdict
from statistics import mode, mean

from ml.dataset import iris, orings, zoo, Majority, Parity, Xor
from ml.losses import MSE, LogLikelihood
from optimization.optimizer import LineSearchOptimizer
from optimization.unconstrained.gradient_descent import GD
from utils import *


class Learner:
    def fit(self, X, y):
        return NotImplementedError

    def predict(self, x):
        return NotImplementedError


def err_ratio(learner, X, y):
    """
    Return the proportion of the examples that are NOT correctly predicted.
    verbose - 0: No output; 1: Output wrong; 2 (or greater): Output correct
    """
    if X.shape[0] == 0:
        return 0.0
    right = 0
    for x, y in zip(X, y):
        if np.isclose(learner.predict(x.reshape((1, -1))), y):
            right += 1
    return 1 - (right / X.shape[0])


def grade_learner(learner, tests):
    """
    Grades the given learner based on how many tests it passes.
    tests is a list with each element in the form: (values, output).
    """
    return mean(int(learner.predict(x) == y) for x, y in tests)


def train_test_split(dataset, start=None, end=None, test_split=None):
    """
    If you are giving 'start' and 'end' as parameters,
    then it will return the testing set from index 'start' to 'end'
    and the rest for training.
    If you give 'test_split' as a parameter then it will return
    test_split * 100% as the testing set and the rest as
    training set.
    """
    examples = dataset.examples
    if test_split is None:
        train = examples[:start] + examples[end:]
        val = examples[start:end]
    else:
        total_size = len(examples)
        val_size = int(total_size * test_split)
        train_size = total_size - val_size
        train = examples[:train_size]
        val = examples[train_size:total_size]

    return train, val


def model_selection(learner, dataset, k=10, trials=1):
    """
    Return the optimal value of size having minimum error on validation set.
    err: a validation error array, indexed by size
    """
    errs = []
    size = 1
    while True:
        err = cross_validation(learner, dataset, size, k, trials)
        # check for convergence provided err_val is not empty
        if err and not np.isclose(err[-1], err, rtol=1e-6):
            best_size = 0
            min_val = np.inf
            i = 0
            while i < size:
                if errs[i] < min_val:
                    min_val = errs[i]
                    best_size = i
                i += 1
            return learner(dataset, best_size)
        errs.append(err)
        size += 1


def cross_validation(learner, dataset, size=None, k=10, trials=1):
    """
    Do k-fold cross_validate and return their mean.
    That is, keep out 1/k of the examples for testing on each of k runs.
    Shuffle the examples first; if trials > 1, average over several shuffles.
    Returns Training error
    """
    k = k or len(dataset.examples)
    if trials > 1:
        trial_errs = 0
        for t in range(trials):
            errs = cross_validation(learner, dataset, size, k, trials)
            trial_errs += errs
        return trial_errs / trials
    else:
        fold_errs = 0
        n = len(dataset.examples)
        examples = dataset.examples
        random.shuffle(dataset.examples)
        for fold in range(k):
            train_data, val_data = train_test_split(dataset, fold * (n // k), (fold + 1) * (n // k))
            dataset.examples = train_data
            h = learner(dataset, size)
            fold_errs += err_ratio(h, dataset, train_data)
            # reverting back to original once test is completed
            dataset.examples = examples
        return fold_errs / k


def leave_one_out(learner, dataset, size=None):
    """Leave one out cross-validation over the dataset."""
    return cross_validation(learner, dataset, size, len(dataset.examples))


def learning_curve(learner, dataset, trials=10, sizes=None):
    if sizes is None:
        sizes = list(range(2, len(dataset.examples) - trials, 2))

    def score(learner, size):
        random.shuffle(dataset.examples)
        return cross_validation(learner, dataset, size, trials)

    return [(size, mean([score(learner, size) for _ in range(trials)])) for size in sizes]


def mean_squared_error(y, y_pred):
    return ((y - y_pred) ** 2).mean()


def r2_score(y, y_pred):
    return 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))


class LinearRegressionLearner(Learner):
    """
    Linear classifier with hard threshold.
    """

    def __init__(self, l_rate=0.01, epochs=1000, optimizer=GD):
        self.l_rate = l_rate
        self.epochs = epochs
        self.optimizer = optimizer

    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if issubclass(self.optimizer, LineSearchOptimizer):
            self.w = self.optimizer(MSE(X, y), np.zeros((X.shape[1], 1)), max_f_eval=self.epochs).minimize()[0]
        else:
            self.w = self.optimizer(MSE(X, y), np.zeros((X.shape[1], 1)),
                                    step_rate=self.l_rate, max_iter=self.epochs).minimize()[0]
        return self

    def predict(self, x):
        return np.dot(x, self.w)


class BinaryLogisticRegressionLearner(Learner):
    """
    Linear classifier with logistic regression.
    """

    def __init__(self, l_rate=0.01, epochs=1000, optimizer=GD):
        self.l_rate = l_rate
        self.epochs = epochs
        self.optimizer = optimizer

    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:, np.newaxis]
        # if issubclass(self.optimizer, LineSearchOptimizer):
        #     self.w = self.optimizer(LogLikelihood(X, y), np.zeros((X.shape[1], 1)),
        #                             max_f_eval=self.epochs).minimize()[0]
        # else:
        #     self.w = self.optimizer(LogLikelihood(X, y), np.zeros((X.shape[1], 1)),
        #                             step_rate=self.l_rate, max_iter=self.epochs).minimize()[0]
        # return self

        from scipy.optimize import minimize
        self.w = minimize(LogLikelihood(X, y).function, np.zeros((X.shape[1], 1)),
                          method='tnc', jac=LogLikelihood(X, y).jacobian).x

    def predict_score(self, x):
        return LogLikelihood.probability(x, self.w[:, np.newaxis])[:, 0]

    def predict(self, x, tol=0.5):
        return (self.predict_score(x) >= tol).astype(int)


class MultiLogisticRegressionLearner(Learner):
    def __init__(self, l_rate=0.01, epochs=1000, optimizer=GD, decision_function='ovr'):
        self.l_rate = l_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.decision_function = decision_function
        self.n_class, self.classifiers = 0, []

    def fit(self, X, y):
        """
        Trains n_class or n_class * (n_class - 1) / 2 classifiers
        according to the training method, ovr or ovo respectively.
        :param X: array of size [n_samples, n_features] holding the training samples
        :param y: array of size [n_samples] holding the class labels
        :return: array of classifiers
        """
        labels = np.unique(y)
        self.n_class = len(labels)
        if self.decision_function == 'ovr':  # one-vs-rest method
            for label in labels:
                y1 = np.array(y)
                y1[y1 != label] = -1.0
                y1[y1 == label] = 1.0
                clf = BinaryLogisticRegressionLearner(self.l_rate, self.epochs, self.optimizer)
                clf.fit(X, y1)
                self.classifiers.append(copy.deepcopy(clf))
        elif self.decision_function == 'ovo':  # use one-vs-one method
            n_labels = len(labels)
            for i in range(n_labels):
                for j in range(i + 1, n_labels):
                    neg_id, pos_id = y == labels[i], y == labels[j]
                    x1, y1 = np.r_[X[neg_id], X[pos_id]], np.r_[y[neg_id], y[pos_id]]
                    y1[y1 == labels[i]] = -1.0
                    y1[y1 == labels[j]] = 1.0
                    clf = BinaryLogisticRegressionLearner(self.l_rate, self.epochs, self.optimizer)
                    clf.fit(x1, y1)
                    self.classifiers.append(copy.deepcopy(clf))
        else:
            return ValueError("Decision function must be either 'ovr' or 'ovo'.")
        return self

    def predict(self, x):
        """
        Predicts the class of a given example according to the training method.
        """
        n_samples = len(x)
        if self.decision_function == 'ovr':  # one-vs-rest method
            assert len(self.classifiers) == self.n_class
            score = np.zeros((n_samples, self.n_class))
            for i in range(self.n_class):
                clf = self.classifiers[i]
                score[:, i] = clf.predict_score(x)
            return np.argmax(score, axis=1)
        elif self.decision_function == 'ovo':  # use one-vs-one method
            assert len(self.classifiers) == self.n_class * (self.n_class - 1) / 2
            vote = np.zeros((n_samples, self.n_class))
            clf_id = 0
            for i in range(self.n_class):
                for j in range(i + 1, self.n_class):
                    res = self.classifiers[clf_id].predict(x)
                    vote[res < 0, i] += 1.0  # negative sample: class i
                    vote[res > 0, j] += 1.0  # positive sample: class j
                    clf_id += 1
            return np.argmax(vote, axis=1)
        else:
            return ValueError("Decision function must be either 'ovr' or 'ovo'.")


def EnsembleLearner(learners):
    """Given a list of learning algorithms, have them vote."""

    def train(dataset):
        predictors = [learner(dataset) for learner in learners]

        def predict(example):
            return mode(predictor(example) for predictor in predictors)

        return predict

    return train


def ada_boost(L, X, y, K):
    target = [x for x in range(X.shape[1])]
    eps = 1 / (2 * X.shape[0])
    w = [1 / X.shape[0]] * X.shape[0]
    h, z = [], []
    for k in range(K):
        h_k = L(w)
        h_k.fit(X, y)
        h.append(h_k)
        error = sum(weight for example, weight in zip(X, w)
                    if example[target] != h_k(example))
        # avoid divide-by-0 from either 0% or 100% error rates
        error = np.clip(error, eps, 1 - eps)
        for j, example in enumerate(X):
            if example[target] == h_k(example):
                w[j] *= error / (1 - error)
        w = normalize(w)
        z.append(np.log((1 - error) / error))
    return weighted_majority(h, z)


def weighted_majority(predictors, weights):
    """Return a predictor that takes a weighted vote."""

    def predict(example):
        return weighted_mode((predictor(example) for predictor in predictors), weights)

    return predict


def weighted_mode(values, weights):
    """
    Return the value with the greatest total weight.
    """
    totals = defaultdict(int)
    for v, w in zip(values, weights):
        totals[v] += w
    return max(totals, key=totals.__getitem__)


def WeightedLearner(unweighted_learner):
    """
    Given a learner that takes just an unweighted dataset, return
    one that takes also a weight for each example.
    """

    def train(dataset, weights):
        return unweighted_learner(replicated_dataset(dataset, weights))

    return train


def replicated_dataset(dataset, weights, n=None):
    """Copy dataset, replicating each example in proportion to its weight."""
    n = n or len(dataset.examples)
    result = copy.copy(dataset)
    result.examples = weighted_replicate(dataset.examples, weights, n)
    return result


def weighted_replicate(seq, weights, n):
    """
    Return n selections from seq, with the count of each element of seq
    proportional to the corresponding weight (filling in fractions randomly).
    """
    assert len(seq) == len(weights)
    weights = normalize(weights)
    wholes = [int(w * n) for w in weights]
    fractions = [(w * n) % 1 for w in weights]
    return (flatten([x] * nx for x, nx in zip(seq, wholes)) +
            weighted_sample_with_replacement(n - sum(wholes), seq, fractions))


def compare(algorithms=None, datasets=None, k=10, trials=1):
    """
    Compare various learners on various datasets using cross-validation.
    Print results as a table.
    """
    # default list of algorithms
    # algorithms = algorithms or [LogisticRegressionLearner, MultiSVM, PerceptronLearner, NeuralNetLearner]

    # default list of datasets
    datasets = datasets or [iris, orings, zoo, Majority(7, 100), Parity(7, 100), Xor(100)]

    print_table([[a.__name__.replace('Learner', '')] + [cross_validation(a, d, k=k, trials=trials) for d in datasets]
                 for a in algorithms], header=[''] + [d.name[0:7] for d in datasets], numfmt='%.2f')
