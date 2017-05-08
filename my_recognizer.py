import warnings
from asl_data import SinglesData

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
