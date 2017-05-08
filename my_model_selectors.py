import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        n = len(self.X) # Number of observations
        best_model = None
        min_score = None

        for i in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(i)
                num_states = model.n_components
                num_zeros_in_transition_matrix = len(model.transmat_.flatten())-np.count_nonzero(model.transmat_)

                # According to https://stats.stackexchange.com/questions/117258/what-are-parameters-in-a-model-and-how-do-i-get-them?rq=1
                p = num_states * (num_states - 1) + num_zeros_in_transition_matrix
                logL = model.score(self.X, self.lengths) # Log likelihood
                bic = -2.0 * logL + p * math.log(n)

                if min_score is None or bic < min_score:
                    min_score = bic
                    best_model = model
            except:
                if self.verbose:
                    print("Failed to compute BIC for " + str(i) + "-state model for " + self.this_word)

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        max_score = None
        M = len(self.hwords)

        for i in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(i)

                logL = model.score(self.X, self.lengths)
                sum_anti_likelihoods = 0.0

                # Calculating likelihoods for all words except self.this_word
                for w in self.words:
                    if w != self.this_word:
                        
                        # Data for each word: w_x are coordinates, w_lengths are number of frames
                        w_x, w_lengths = self.hwords[w]
                        sum_anti_likelihoods = sum_anti_likelihoods + model.score(w_x, w_lengths)

                average_anti_likelihood = sum_anti_likelihoods / (M-1)
                dic = logL - average_anti_likelihood

                if max_score is None or dic > max_score:
                    max_score = dic
                    best_model = model
            except:
                if self.verbose:
                    print("Failed to compute DIC for " + str(i) + "-state model for " + self.this_word)

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        split_method = KFold(n_splits=2)
        best_model = None
        max_score = None

        for i in range(self.min_n_components, self.max_n_components + 1):
            try:
                n_folds = 0
                sum_likelihoods = 0

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    model = self.base_model(i)

                    test_x, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    logL = model.score(test_x, test_lengths)

                    sum_likelihoods = sum_likelihoods + logL
                    n_folds += 1

                avg_likelihood = sum_likelihoods / n_folds

                if max_score is None or avg_likelihood > max_score:
                    max_score = avg_likelihood
                    best_model = model
            except:
                if self.verbose:
                    print("Failed to perform cross-validation for " + str(i) + "-state model for " + self.this_word)

        return best_model
