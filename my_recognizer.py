import warnings
from asl_data import SinglesData
from asl_utils import show_errors

import numpy as np
import pandas as pd
from asl_data import AslDb
from hmmlearn.hmm import GaussianHMM
from my_model_selectors import SelectorConstant
from my_model_selectors import SelectorBIC
from my_model_selectors import SelectorDIC
from my_model_selectors import SelectorCV


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    all_x_lengths = test_set.get_all_Xlengths()
    keys = list(all_x_lengths.keys())
    keys.sort()

    for key in keys:
        x, lengths = all_x_lengths[key]
        scores = {}
        max_score = None
        best_guess = ""

        for word in models:
            model = models[word]

            try:
                score = model.score(x, lengths)
                scores[word] = score

                if max_score is None or score > max_score:
                    max_score = score
                    best_guess = word
            except:
                scores[word] = float("-inf")

        probabilities.append(scores)
        guesses.append(best_guess)

    return probabilities, guesses


def train_a_word(word, num_hidden_states, features):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    training = asl.build_training(features)
    X, lengths = training.get_word_Xlengths(word)
    model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
    logL = model.score(X, lengths)
    return model, logL

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

if __name__ == "__main__":
    asl = AslDb()

    asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
    asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
    asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
    asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

    features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']

    df_means = asl.df.groupby('speaker').mean()

    asl.df['left-x-mean'] = asl.df['speaker'].map(df_means['left-x'])
    asl.df['left-y-mean'] = asl.df['speaker'].map(df_means['left-y'])
    asl.df['right-x-mean'] = asl.df['speaker'].map(df_means['right-x'])
    asl.df['right-y-mean'] = asl.df['speaker'].map(df_means['right-y'])

    df_std = asl.df.groupby('speaker').std()

    asl.df['left-x-std'] = asl.df['speaker'].map(df_std['left-x'])
    asl.df['left-y-std'] = asl.df['speaker'].map(df_std['left-y'])
    asl.df['right-x-std'] = asl.df['speaker'].map(df_std['right-x'])
    asl.df['right-y-std'] = asl.df['speaker'].map(df_std['right-y'])

    features_norm = ['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']

    asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['right-x-mean']) / asl.df['right-x-std']
    asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['right-y-mean']) / asl.df['right-y-std']
    asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['left-x-mean']) / asl.df['left-x-std']
    asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['left-y-mean']) / asl.df['left-y-std']

    features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

    asl.df['polar-rr'] = np.hypot(asl.df['grnd-rx'], asl.df['grnd-ry'])
    asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
    asl.df['polar-lr'] = np.hypot(asl.df['grnd-lx'], asl.df['grnd-ly'])
    asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])

    features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

    asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
    asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
    asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
    asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)

    df_minimums = asl.df.groupby('speaker').min()
    asl.df['polar-rr-min'] = asl.df['speaker'].map(df_minimums['polar-rr'])
    asl.df['polar-rtheta-min'] = asl.df['speaker'].map(df_minimums['polar-rtheta'])
    asl.df['polar-lr-min'] = asl.df['speaker'].map(df_minimums['polar-lr'])
    asl.df['polar-ltheta-min'] = asl.df['speaker'].map(df_minimums['polar-ltheta'])

    asl.df['grnd-ry-min'] = asl.df['speaker'].map(df_minimums['grnd-ry'])
    asl.df['grnd-rx-min'] = asl.df['speaker'].map(df_minimums['grnd-rx'])
    asl.df['grnd-ly-min'] = asl.df['speaker'].map(df_minimums['grnd-ly'])
    asl.df['grnd-lx-min'] = asl.df['speaker'].map(df_minimums['grnd-lx'])

    df_maximums = asl.df.groupby('speaker').max()
    asl.df['polar-rr-max'] = asl.df['speaker'].map(df_maximums['polar-rr'])
    asl.df['polar-rtheta-max'] = asl.df['speaker'].map(df_maximums['polar-rtheta'])
    asl.df['polar-lr-max'] = asl.df['speaker'].map(df_maximums['polar-lr'])
    asl.df['polar-ltheta-max'] = asl.df['speaker'].map(df_maximums['polar-ltheta'])

    asl.df['grnd-ry-max'] = asl.df['speaker'].map(df_maximums['grnd-ry'])
    asl.df['grnd-rx-max'] = asl.df['speaker'].map(df_maximums['grnd-rx'])
    asl.df['grnd-ly-max'] = asl.df['speaker'].map(df_maximums['grnd-ly'])
    asl.df['grnd-lx-max'] = asl.df['speaker'].map(df_maximums['grnd-lx'])

    asl.df['scaled-rr'] = (asl.df['polar-rr'] - asl.df['polar-rr-min']) / (asl.df['polar-rr-max'] - asl.df['polar-rr-min'])
    asl.df['scaled-rtheta'] = (asl.df['polar-rtheta'] - asl.df['polar-rtheta-min']) / (asl.df['polar-rtheta-max'] - asl.df['polar-rtheta-min'])
    asl.df['scaled-lr'] = (asl.df['polar-lr'] - asl.df['polar-lr-min']) / (asl.df['polar-lr-max'] - asl.df['polar-lr-min'])
    asl.df['scaled-ltheta'] = (asl.df['polar-ltheta'] - asl.df['polar-ltheta-min']) / (asl.df['polar-ltheta-max'] - asl.df['polar-ltheta-min'])

    asl.df['scaled-grnd-ry'] = (asl.df['grnd-ry'] - asl.df['grnd-ry-min']) / (asl.df['grnd-ry-max'] - asl.df['grnd-ry-min'])
    asl.df['scaled-grnd-rx'] = (asl.df['grnd-rx'] - asl.df['grnd-rx-min']) / (asl.df['grnd-rx-max'] - asl.df['grnd-rx-min'])
    asl.df['scaled-grnd-ly'] = (asl.df['grnd-ly'] - asl.df['grnd-ly-min']) / (asl.df['grnd-ly-max'] - asl.df['grnd-ly-min'])
    asl.df['scaled-grnd-lx'] = (asl.df['grnd-lx'] - asl.df['grnd-lx-min']) / (asl.df['grnd-lx-max'] - asl.df['grnd-lx-min'])

    features_custom = [
        'norm-rr', 'norm-rtheta', 'norm-lr', 'norm-ltheta',
        'delta-norm-rr', 'delta-norm-rtheta', 'delta-norm-lr', 'delta-norm-ltheta'
    ]

    df_means = asl.df.groupby('speaker').mean()
    asl.df['polar-rr-mean'] = asl.df['speaker'].map(df_means['polar-rr'])
    asl.df['polar-rtheta-mean'] = asl.df['speaker'].map(df_means['polar-rtheta'])
    asl.df['polar-lr-mean'] = asl.df['speaker'].map(df_means['polar-lr'])
    asl.df['polar-ltheta-mean'] = asl.df['speaker'].map(df_means['polar-ltheta'])

    df_std = asl.df.groupby('speaker').std()
    asl.df['polar-rr-std'] = asl.df['speaker'].map(df_std['polar-rr'])
    asl.df['polar-rtheta-std'] = asl.df['speaker'].map(df_std['polar-rtheta'])
    asl.df['polar-lr-std'] = asl.df['speaker'].map(df_std['polar-lr'])
    asl.df['polar-ltheta-std'] = asl.df['speaker'].map(df_std['polar-ltheta'])

    asl.df['norm-rr'] = (asl.df['polar-rr'] - asl.df['polar-rr-mean']) / asl.df['polar-rr-std']
    asl.df['norm-rtheta'] = (asl.df['polar-rtheta'] - asl.df['polar-rtheta-mean']) / asl.df['polar-rtheta-std']
    asl.df['norm-lr'] = (asl.df['polar-lr'] - asl.df['polar-lr-mean']) / asl.df['polar-lr-std']
    asl.df['norm-ltheta'] = (asl.df['polar-ltheta'] - asl.df['polar-ltheta-mean']) / asl.df['polar-ltheta-std']

    asl.df['delta-norm-rr'] = asl.df['norm-rr'].diff().fillna(0)
    asl.df['delta-norm-rtheta'] = asl.df['norm-rtheta'].diff().fillna(0)
    asl.df['delta-norm-lr'] = asl.df['norm-lr'].diff().fillna(0)
    asl.df['delta-norm-ltheta'] = asl.df['norm-ltheta'].diff().fillna(0)

    scaled_ground_coordinates = ['scaled-grnd-ry', 'scaled-grnd-rx', 'scaled-grnd-ly', 'scaled-grnd-lx']
    normalized_polar_coordinates = ['norm-rr', 'norm-rtheta', 'norm-lr', 'norm-ltheta']
    scaled_polar_coordinates = ['scaled-rr', 'scaled-rtheta', 'scaled-lr', 'scaled-ltheta']
    normalized_polar_deltas = ['delta-norm-rr', 'delta-norm-rtheta', 'delta-norm-lr', 'delta-norm-ltheta']
    scaled_polar_deltas = ['delta-scaled-rr', 'delta-scaled-rtheta', 'delta-scaled-lr', 'delta-scaled-ltheta']

    # An interesting part starts here

    features_map = {#'features_ground' : features_ground,
                    #'features_norm' : features_norm,
                    #'features_delta' : features_delta,
                    #'normalized_polar_coordinates' : normalized_polar_coordinates,
                    # 'scaled_polar_coordinates' : scaled_polar_coordinates,
                    'scaled_ground_coordinates' : scaled_ground_coordinates
                    #'features_polar' : features_polar,
                    #'normalized_polar_deltas' : normalized_polar_deltas,
                    # 'scaled_polar_deltas' : scaled_polar_deltas
    }

    selectors_map = {'selector_bic' : SelectorBIC,
                     'selector_dic' : SelectorDIC,
                     'selector_cv' : SelectorCV }

    for f in features_map:
        for s in selectors_map:
            print("Evaluating " + s + " and " + f + " (  " + f + "_" + s + ".txt  )\n")

            features = features_map[f]
            model_selector = selectors_map[s]

            print("Training...\n")
            models = train_all_words(features, model_selector)
            test_set = asl.build_test(features)

            print("Recognizing...\n")
            probabilities, guesses = recognize(models, test_set)

            show_errors(guesses, test_set)

            print("\n\n")


