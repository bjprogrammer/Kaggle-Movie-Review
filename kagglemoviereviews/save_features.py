''' 
  The main goal of the program is to illustrate the use of the writeFeatureSets function,
    which can be used for other data represented as feature sets to write a file
    for weka or sklearn to read for classification
  this program uses the movie reviews from the original Pang and Lee data,
    including 1000 positive and 1000 negative reviews
  for each review, we define what the NLTK calls a feature set,
    which consists of a feature dictionary, a python dictionary mapping feature names to values
    paired together with the gold label of that review
  the function writeFeatureSets will write that data structure to a csv file that 
    either weka or sklearn can read
 
  Usage:  python save_features.py <filename or path>
'''

import sys
import nltk
from nltk.corpus import movie_reviews
import random

# for testing, allow different sizes for word features
vocab_size = 100

# Function writeFeatureSets:
# takes featuresets defined in the nltk and convert them to weka input csv file
#    any feature value in the featuresets should not contain ",", "'" or " itself
#    and write the file to the outpath location
#    outpath should include the name of the csv file
def writeFeatureSets(featuresets, outpath):
    # open outpath for writing
    f = open(outpath, 'w')
    # get the feature names from the feature dictionary in the first featureset
    featurenames = featuresets[0][0].keys()
    # create the first line of the file as comma separated feature names
    #    with the word class as the last feature name
    featurenameline = ''
    for featurename in featurenames:
        # replace forbidden characters with text abbreviations
        featurename = featurename.replace(',','CM')
        featurename = featurename.replace("'","DQ")
        featurename = featurename.replace('"','QU')
        featurenameline += featurename + ','
    featurenameline += 'class'
    # write this as the first line in the csv file
    f.write(featurenameline)
    f.write('\n')
    # convert each feature set to a line in the file with comma separated feature values,
    # each feature value is converted to a string 
    #   for booleans this is the words true and false
    #   for numbers, this is the string with the number
    for featureset in featuresets:
        featureline = ''
        for key in featurenames:
          try:
            featureline += str(featureset[0].get(key,[])) + ','
          except KeyError:
            continue
        if featureset[1] == 0:
          featureline += str("strongly negative")
        elif featureset[1] == 1:
          featureline += str("slightly negative")
        elif featureset[1] == 2:
          featureline += str("neutral")
        elif featureset[1] == 3:
          featureline += str("slightly positive")
        elif featureset[1] == 4:
          featureline += str("strongly positive")
        # write each feature set values to the file
        f.write(featureline)
        f.write('\n')
    f.close()

# define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'contains(keyword)' and is true or false depending
# on whether that keyword is in the document
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features

# Main program to produce movie review feature sets in order to show how to use 
#   the writeFeatureSets function
if __name__ == '__main__':
    # Make a list of command line arguments, omitting the [0] element
    # which is the script itself.
    args = sys.argv[1:]
    if not args:
        print ('usage: python save_features.py [file]')
        sys.exit(1)
    outpath = args[0]
    
    # for each document in movie_reviews, get its words and category (positive/negative)
    documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)

    # get all words from all movie_reviews and put into a frequency distribution
    #   note lowercase, but no stemming or stopwords
    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

    # get the most frequently appearing keywords in the corpus
    word_items = all_words.most_common(vocab_size)
    word_features = [word for (word, freq) in word_items]

    # get features sets for a document, including keyword features and category feature
    featuresets = [(document_features(d, word_features), c) for (d, c) in documents]

    # write the feature sets to the csv file
    writeFeatureSets(featuresets, outpath)

    print ('Wrote movie review features to:', outpath)
