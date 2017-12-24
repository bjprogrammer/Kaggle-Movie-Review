# -*- coding: utf-8 -*-

# This program contains a function to get positive and negative emotion words from
#   LIWC, the Linguistic Inquiry and Word Count resource from James Pennebaker
#   This resouce was obtained for research and educational purposes and should
#   not be used in products or for commercial gain without proper licensing.

import os
import sys

# returns two lists:  words in positive emotion class and
#		      words in negative emotion class
def read_words():
  poslist = []
  neglist = []

  flexicon = open('/Users/bobby/Downloads/kagglemoviereviews/kagglemoviereviews/SentimentLexicons/liwcdic2007.dic', encoding='latin1')
  # read all LIWC words from file
  wordlines = [line.strip() for line in flexicon]
  # each line has a word or a stem followed by * and numbers of the word classes it is in
  # word class 126 is positive emotion and 127 is negative emotion
  for line in wordlines:
    if not line == '':
      items = line.split()
      word = items[0]
      classes = items[1:]
      for c in classes:
        if c == '126':
          poslist.append( word )
        if c == '127':
          neglist.append( word )
  return (poslist, neglist)

# test to see if a word is on the list
#   using a prefix test if the word is a stem with an *
# returns True or False
def isPresent(word, emotionlist):
  isFound = False
  # loop over all elements of list
  for emotionword in emotionlist:
    # test if a word or a stem
    if not emotionword[-1] == '*':
      # it's a word!
      # when a match is found, can quit the loop with True
      if word == emotionword:
        isFound = True
        break
    else:
      # it's a stem!
      # when a match is found, can quit the loop with True
      if word.startswith(emotionword[0:-1]):
        isFound = True
        break
  # end of loop
  return isFound

# for testing purposes, run the file with no command line arguments
# the main prints all positive and negative words
# and tests some words
if __name__=='__main__':
  (poslist, neglist) = read_words()
  print ("Positive Words", len(poslist), "Negative Words", len(neglist))
  print (poslist)
  print (neglist)
  words = ['ache', 'crashed', 'worsen', 'worthless', 'nice']
  for word in words:
    print (word, isPresent(word, poslist), isPresent(word, neglist))

        
    



