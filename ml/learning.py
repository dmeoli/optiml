import copy
import heapq
from collections import defaultdict
from statistics import stdev, mode

from ml.activations import Sigmoid
from ml.datasets import iris, orings, zoo, Majority, Parity, Xor
from utils import *


class DataSet:
    """
    A data set for a machine learning problem. It has the following fields:

    d.examples   A list of examples. Each one is a list of attribute values.
    d.attrs      A list of integers to index into an example, so example[attr]
                 gives a value. Normally the same as range(len(d.examples[0])).
    d.attr_names Optional list of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.inputs     The list of attrs without the target.
    d.values     A list of lists: each sublist is the set of possible
                 values for the corresponding attribute. If initially None,
                 it is computed from the known examples by self.set_problem.
                 If not None, an erroneous value raises ValueError.
    d.distance   A function from a pair of examples to a non-negative number.
                 Should be symmetric, etc. Defaults to mean_boolean_error
                 since that can handle any field types.
    d.name       Name of the data set (for output display only).
    d.source     URL or other source where the data came from.
    d.exclude    A list of attribute indexes to exclude from d.inputs. Elements
                 of this list can either be integers (attrs) or attr_names.

    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs.
    """

    def __init__(self, examples=None, attrs=None, attr_names=None, target=-1, inputs=None,
                 values=None, distance=mean_boolean_error, name='', source='', exclude=()):
        """
        Accepts any of DataSet's fields. Examples can also be a
        string or file from which to parse examples using parse_csv.
        Optional parameter: exclude, as documented in .set_problem().
        """
        self.name = name
        self.source = source
        self.values = values
        self.distance = distance
        self.got_values_flag = bool(values)

        # initialize .examples from string or list or data directory
        if isinstance(examples, str):
            self.examples = parse_csv(examples)
        elif examples is None:
            self.examples = parse_csv(open_data(name + '.csv').read())
        else:
            self.examples = examples

        # attrs are the indices of examples, unless otherwise stated.
        if self.examples is not None and attrs is None:
            attrs = list(range(len(self.examples[0])))

        self.attrs = attrs

        # initialize .attr_names from string, list, or by default
        if isinstance(attr_names, str):
            self.attr_names = attr_names.split()
        else:
            self.attr_names = attr_names or attrs
        self.set_problem(target, inputs=inputs, exclude=exclude)

    def set_problem(self, target, inputs=None, exclude=()):
        """
        Set (or change) the target and/or inputs.
        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not use in inputs. Attributes can be -n .. n, or an attr_name.
        Also computes the list of possible values, if that wasn't done yet.
        """
        self.target = self.attr_num(target)
        exclude = list(map(self.attr_num, exclude))
        if inputs:
            self.inputs = remove_all(self.target, inputs)
        else:
            self.inputs = [a for a in self.attrs if a != self.target and a not in exclude]
        if not self.values:
            self.update_values()
        self.check_me()

    def check_me(self):
        """Check that my fields make sense."""
        assert len(self.attr_names) == len(self.attrs)
        assert self.target in self.attrs
        assert self.target not in self.inputs
        assert set(self.inputs).issubset(set(self.attrs))
        if self.got_values_flag:
            # only check if values are provided while initializing DataSet
            list(map(self.check_example, self.examples))

    def add_example(self, example):
        """Add an example to the list of examples, checking it first."""
        self.check_example(example)
        self.examples.append(example)

    def check_example(self, example):
        """Raise ValueError if example has any invalid values."""
        if self.values:
            for a in self.attrs:
                if example[a] not in self.values[a]:
                    raise ValueError('Bad value {} for attribute {} in {}'
                                     .format(example[a], self.attr_names[a], example))

    def attr_num(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attr_names.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr

    def update_values(self):
        self.values = list(map(unique, zip(*self.examples)))

    def sanitize(self, example):
        """Return a copy of example, with non-input attributes replaced by None."""
        return [attr_i if i in self.inputs else None for i, attr_i in enumerate(example)]

    def classes_to_numbers(self, classes=None):
        """Converts class names to numbers."""
        if not classes:
            # if classes were not given, extract them from values
            classes = sorted(self.values[self.target])
        for item in self.examples:
            item[self.target] = classes.index(item[self.target])

    def remove_examples(self, value=''):
        """Remove examples that contain given value."""
        self.examples = [x for x in self.examples if value not in x]
        self.update_values()

    def split_values_by_classes(self):
        """Split values into buckets according to their class."""
        buckets = defaultdict(lambda: [])
        target_names = self.values[self.target]

        for v in self.examples:
            item = [a for a in v if a not in target_names]  # remove target from item
            buckets[v[self.target]].append(item)  # add item to bucket of its class

        return buckets

    def find_means_and_deviations(self):
        """
        Finds the means and standard deviations of self.dataset.
        means     : a dictionary for each class/target. Holds a list of the means
                    of the features for the class.
        deviations: a dictionary for each class/target. Holds a list of the sample
                    standard deviations of the features for the class.
        """
        target_names = self.values[self.target]
        feature_numbers = len(self.inputs)

        item_buckets = self.split_values_by_classes()

        means = defaultdict(lambda: [0] * feature_numbers)
        deviations = defaultdict(lambda: [0] * feature_numbers)

        for t in target_names:
            # find all the item feature values for item in class t
            features = [[] for _ in range(feature_numbers)]
            for item in item_buckets[t]:
                for i in range(feature_numbers):
                    features[i].append(item[i])

            # calculate means and deviations fo the class
            for i in range(feature_numbers):
                means[t][i] = mean(features[i])
                deviations[t][i] = stdev(features[i])

        return means, deviations

    def __repr__(self):
        return '<DataSet({}): {:d} examples, {:d} attributes>'.format(self.name, len(self.examples), len(self.attrs))


def parse_csv(input, delim=','):
    r"""
    Input is a string consisting of lines, each line has comma-delimited
    fields. Convert this into a list of lists. Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.
    """
    lines = [line for line in input.splitlines() if line.strip()]
    return [list(map(num_or_str, line.split(delim))) for line in lines]


def err_ratio(predict, dataset, examples=None, verbose=0):
    """
    Return the proportion of the examples that are NOT correctly predicted.
    verbose - 0: No output; 1: Output wrong; 2 (or greater): Output correct
    """
    examples = examples or dataset.examples
    if len(examples) == 0:
        return 0.0
    right = 0
    for example in examples:
        desired = example[dataset.target]
        output = predict(dataset.sanitize(example))
        if output == desired:
            right += 1
            if verbose >= 2:
                print('   OK: got {} for {}'.format(desired, example))
        elif verbose:
            print('WRONG: got {}, expected {} for {}'.format(output, desired, example))
    return 1 - (right / len(examples))


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


def NearestNeighborLearner(dataset, k=1):
    """k-NearestNeighbor: the k nearest neighbors vote."""

    def predict(example):
        """Find the k closest items, and have them vote for the best."""
        best = heapq.nsmallest(k, ((dataset.distance(e, example), e) for e in dataset.examples))
        return mode(e[dataset.target] for (d, e) in best)

    return predict


def random_weights(min_value, max_value, num_weights):
    return [random.uniform(min_value, max_value) for _ in range(num_weights)]


def LinearLearner(dataset, learning_rate=0.01, epochs=100):
    """
    Linear classifier with hard threshold.
    """
    idx_i = dataset.inputs
    idx_t = dataset.target
    examples = dataset.examples
    num_examples = len(examples)

    # X transpose
    X_col = [dataset.values[i] for i in idx_i]  # vertical columns of X

    # add dummy
    ones = [1 for _ in range(len(examples))]
    X_col = [ones] + X_col

    # initialize random weights
    num_weights = len(idx_i) + 1
    w = random_weights(min_value=-0.5, max_value=0.5, num_weights=num_weights)

    for epoch in range(epochs):
        err = []
        # pass over all examples
        for example in examples:
            x = [1] + example
            y = dot_product(w, x)
            t = example[idx_t]
            err.append(t - y)

        # update weights
        for i in range(len(w)):
            w[i] = w[i] + learning_rate * (dot_product(err, X_col[i]) / num_examples)

    def predict(example):
        x = [1] + example
        return dot_product(w, x)

    return predict


def LogisticLinearLeaner(dataset, learning_rate=0.01, epochs=100):
    """
    Linear classifier with logistic regression.
    """
    idx_i = dataset.inputs
    idx_t = dataset.target
    examples = dataset.examples
    num_examples = len(examples)

    # X transpose
    X_col = [dataset.values[i] for i in idx_i]  # vertical columns of X

    # add dummy
    ones = [1 for _ in range(len(examples))]
    X_col = [ones] + X_col

    # initialize random weights
    num_weights = len(idx_i) + 1
    w = random_weights(min_value=-0.5, max_value=0.5, num_weights=num_weights)

    for epoch in range(epochs):
        err = []
        h = []
        # pass over all examples
        for example in examples:
            x = [1] + example
            y = Sigmoid().function(dot_product(w, x))
            h.append(Sigmoid().derivative(y))
            t = example[idx_t]
            err.append(t - y)

        # update weights
        for i in range(len(w)):
            buffer = [x * y for x, y in zip(err, h)]
            w[i] = w[i] + learning_rate * (dot_product(buffer, X_col[i]) / num_examples)

    def predict(example):
        x = [1] + example
        return Sigmoid().function(dot_product(w, x))

    return predict


def EnsembleLearner(learners):
    """Given a list of learning algorithms, have them vote."""

    def train(dataset):
        predictors = [learner(dataset) for learner in learners]

        def predict(example):
            return mode(predictor(example) for predictor in predictors)

        return predict

    return train


def ada_boost(dataset, L, K):
    examples, target = dataset.examples, dataset.target
    n = len(examples)
    eps = 1 / (2 * n)
    w = [1 / n] * n
    h, z = [], []
    for k in range(K):
        h_k = L(dataset, w)
        h.append(h_k)
        error = sum(weight for example, weight in zip(examples, w) if example[target] != h_k(example))
        # avoid divide-by-0 from either 0% or 100% error rates
        error = np.clip(error, eps, 1 - eps)
        for j, example in enumerate(examples):
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
    [Page 749 footnote 14]
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
    Return n selections from seq, with the count of each element of
    seq proportional to the corresponding weight (filling in fractions
    randomly).
    """
    assert len(seq) == len(weights)
    weights = normalize(weights)
    wholes = [int(w * n) for w in weights]
    fractions = [(w * n) % 1 for w in weights]
    return (flatten([x] * nx for x, nx in zip(seq, wholes)) +
            weighted_sample_with_replacement(n - sum(wholes), seq, fractions))


def flatten(seqs):
    return sum(seqs, [])


def compare(algorithms=None, datasets=None, k=10, trials=1):
    """
    Compare various learners on various datasets using cross-validation.
    Print results as a table.
    """
    # default list of algorithms
    algorithms = algorithms or [NearestNeighborLearner]

    # default list of datasets
    datasets = datasets or [iris, orings, zoo, Majority(7, 100), Parity(7, 100), Xor(100)]

    print_table([[a.__name__.replace('Learner', '')] + [cross_validation(a, d, k=k, trials=trials) for d in datasets]
                 for a in algorithms], header=[''] + [d.name[0:7] for d in datasets], numfmt='%.2f')
