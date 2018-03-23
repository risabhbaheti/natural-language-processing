#!/bin/python
from nltk.stem import WordNetLemmatizer

'''st = StanfordNERTagger("/home/risabh/work/softwares/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz",
                       "/home/risabh/work/softwares/stanford-ner/stanford-ner.jar",
                       encoding='utf-8')
'''
wordnet_lemmatizer = WordNetLemmatizer()
def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    pass

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")
    
    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    
    if word.istitle():
        ftrs.append("IS_TITLE")
    if word[1:].lower() != word[1:]:
        ftrs.append("CAPITAL_INSIDE")
    
    if len(word)>0:
        ftrs.append(word.lower()[-1:])
    if len(word)>1:
        ftrs.append(word.lower()[-2:])
    if len(word)>2:
        ftrs.append(word.lower()[-3:])
    if word.startswith('@') or word.startswith('http') or word.startswith('#'):
        ftrs.append("IS_X")              
    ftrs.append(wordnet_lemmatizer.lemmatize(word.lower()))


    # previous/next word feats
    if add_neighs:
        val = False
        if i > 0:
            for pf in token2features(sent, i-1, val):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, val):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "Christine", "loves", "food", "and", "pizza" ],
    [ "Jessica", "loves", "food", "and", "pizza" ],
    [ "Aaron", "loves", "food", "and", "pizza" ],
    [ "Christine", "loves", "food", "and", "pizza" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
