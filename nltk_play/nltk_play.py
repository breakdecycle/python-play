##### Natural Language Processing #####
# Source:
# https://www.youtube.com/playlist?list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

############### Part 1 ###############
# Tokenizing

def part_1():
    from nltk.tokenize import sent_tokenize, word_tokenize

    txt = 'Hello there Mr. Lee, How are you doing today? The weather is great today'

    txt_stok = sent_tokenize(txt)
    txt_wtok = word_tokenize(txt)
    print (txt_stok)
    print (txt_wtok)

    print('\n\n')

# Comment in/out to run this part:
# part_1()

############### Part 2 ###############
# Stop words

def part_2():
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    txt2 = 'This is an example showing off stop word filtration.'

    stop_words = set(stopwords.words("english"))

    # What are stop words?
    print (stop_words)

    words = txt2_wtok = word_tokenize(txt2)
    filtered_sentence1 = []

    # Filter and print the stop words.
    for w in words:
        if w not in stop_words:
            filtered_sentence1.append(w)
    # OR similarly
    filtered_sentence2 = [w for w in words if not w in stop_words]

    print(filtered_sentence1)
    print(filtered_sentence2)

    print('\n\n')

# Comment in/out to run this part:
# part_2()

############### Part 3 ###############
# (Root) Stemming
#
# I was taking a ride in the car.
# I was riding in the car.
# ^ Same meaning but the verb is using -ing form.
# Stemming helps remove the redundancy of having too many words conveying the same meaning.

def part_3():
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    ps = PorterStemmer()

    wrd1 = ['python', 'pythoner', 'pythoning', 'pythoned', 'pythonly']

    # Print the root stem of the words.
    for w in wrd1:
        print (ps.stem(w))

    txt3 = 'It is very important to be pythonly while you are pythoning \
            with python. All pythoners have pythoned poorly at least once.'

    wrd2 = word_tokenize(txt3)

    print('\nNext Example... \n')
    for w in wrd2:
        print (ps.stem(w))

    print('\n\n')

# Comment in/out to run this part:
# part_3()

############### Part 4 ###############
# Part of Speech Tagging

def part_4():
    import nltk
    from nltk.corpus import state_union
    from nltk.tokenize import PunktSentenceTokenizer

    train_txt = state_union.raw("2006-GWBush.txt")
    txt4 = state_union.raw("2006-GWBush.txt")

    custom_sent_tokenizer = PunktSentenceTokenizer(train_txt)

    txt4_tok = custom_sent_tokenizer.tokenize(txt4)

    def process_content():
        try:
            for i in txt4_tok:
                words = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(words)
                print (tagged) # Comment out to not flood terminal
        except Exception as e:
            print (str(e))

    # Comment out to not run this process below:
    process_content()

    print('\n\n')

# Comment in/out to run this part:
# part_4()

############### Part 5 ###############
# Chunking

def part_5():
    import nltk
    from nltk.corpus import state_union
    from nltk.tokenize import PunktSentenceTokenizer

    train_txt = state_union.raw("2006-GWBush.txt")
    txt4 = state_union.raw("2006-GWBush.txt")

    custom_sent_tokenizer = PunktSentenceTokenizer(train_txt)

    txt4_tok = custom_sent_tokenizer.tokenize(txt4)

    def process_content_p5():
        try:
            for i in txt4_tok:
                words = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(words)

                # Below uses RegEx to find specific type of words as chunks.
                chunkGram = r"""Chunk: {<RB.?>*<VB.?><NNP>+<NN>?}"""

                # Parsing this chunk
                chunkParser = nltk.RegexpParser(chunkGram)
                chunked = chunkParser.parse(tagged)

                # Can just print out each step, but too messy...
                # print (chunked) # Comment out if needed. 
                # ... So use Matplotlib to visualise:
                chunked.draw() # Comment out if needed. 
        except Exception as e:
            print (str(e))
        
    # Comment out to not run this process below:
    process_content_p5()

    print('\n\n')

# Comment in/out to run this part:
# part_5()

############### Part 6 ###############
# Chinking (ie. excludes/remove something from a chuck)

def part_6():
    import nltk
    from nltk.corpus import state_union
    from nltk.tokenize import PunktSentenceTokenizer

    train_txt = state_union.raw("2006-GWBush.txt")
    txt4 = state_union.raw("2006-GWBush.txt")

    custom_sent_tokenizer = PunktSentenceTokenizer(train_txt)

    txt4_tok = custom_sent_tokenizer.tokenize(txt4)

    def process_content_p6():
        try:
            for i in txt4_tok:
                words = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(words)

                # Below uses RegEx to find specific type of words as chunks.
                chunkGram = r"""Chunk: {<.*>+}                  # Chunks everything... except for...
                                        }<VB.?|IN|DT|TO>{"""    # }{ defines a chink

                # Parsing this chunk
                chunkParser = nltk.RegexpParser(chunkGram)
                chunked = chunkParser.parse(tagged)

                # Can just print out each step, but too messy...
                # print (chunked) # Comment out if needed. 
                # ... So use Matplotlib to visualise:
                chunked.draw() # Comment out if needed. 
        except Exception as e:
            print (str(e))
        
    # Comment out to not run this process below:
    process_content_p6()

    print('\n\n')

# Comment in/out to run this part:
# part_6()

############### Part 7 ###############
# Named Entity Recognition
# To find out what the subjects of discussion are.

def part_7():
    import nltk
    from nltk.corpus import state_union
    from nltk.tokenize import PunktSentenceTokenizer

    train_txt = state_union.raw("2006-GWBush.txt")
    txt4 = state_union.raw("2006-GWBush.txt")

    custom_sent_tokenizer = PunktSentenceTokenizer(train_txt)

    txt4_tok = custom_sent_tokenizer.tokenize(txt4)

    def process_content_p7():
        try:
            for i in txt4_tok:
                words = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(words)

                nameEnt = nltk.ne_chunk(tagged, binary = True) # Ent = entity

                nameEnt.draw()

        except Exception as e:
            print (str(e))
        
    # Comment out to not run this process below:
    process_content_p7()

    print('\n\n')

# Comment in/out to run this part:
# part_7()

############### Part 8 ###############
# Lemmatizing 
# Similar to Stemming but difference is that doesn't create
# non-existent words.
# Lemmatizer my be better or useful than Stemming, but stemming has its use case too.

def part_8():
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    print (lemmatizer.lemmatize('cats'))
    print (lemmatizer.lemmatize('cacti'))
    print (lemmatizer.lemmatize('geese'))
    print (lemmatizer.lemmatize('rocks'))
    print (lemmatizer.lemmatize('python'))

    print (lemmatizer.lemmatize('better'))
    print (lemmatizer.lemmatize('better', pos='a')) # a = adjective

    print (lemmatizer.lemmatize('best', pos="a")) 

    print (lemmatizer.lemmatize('run'))
    print (lemmatizer.lemmatize('run', 'v')) 

    print('\n\n')

# Comment in/out to run this part:
# part_8()

############### Part 9 ###############
# NLTK Corpora 
# Loading and playing arund with corpora...
# Corpora is just a body of texts. 
# Generally, corpora are grouped by some sort of defining characteristic.


def part_9():
    # Find the directory where the module/library is located in.
    # import nltk
    # print (nltk.__file__)
    #
    # Take note of 'Common locations'
    #
    # For Windows, type '%appdata%' in Explorer, then see '...\nltk_data\corpora'

    from nltk.corpus import gutenberg
    from nltk.tokenize import sent_tokenize

    sample = gutenberg.raw("bible-kjv.txt")
    tok = sent_tokenize(sample)

    print (tok[5:15])
    
    print('\n\n')

# Comment in/out to run this part:
# part_9()


############### Part 10 ###############
# WordNet - among the largest corpora
# Loading and playing arund with this corpora...

def part_10():
    from nltk.corpus import wordnet

    syns = wordnet.synsets("program") # syns = synonym sets
    def syns_play():
        print(syns) # print list of synonym sets
        print()

        # synset:
        print (syns[0].name()) 
        
        # Just the word:
        print (syns[0].lemmas()[0].name())

        # Definition
        print (syns[0].definition())

        # Examples
        print (syns[0].examples())

        print()
    
    # Comment in/out if needed:
    # syns_play()

    # For synonyms and antonyms
    # Simple example
    def synant_sim():
        synonyms = []
        antonyms = []

        for syn in wordnet.synsets('good'):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
        
        print (set(synonyms))
        print ()
        print (set(antonyms))

    # Comment in/out if needed:
    # synant_sim()
        
    # For synonyms and antonyms
    # Detailed example - all the possible lemmas
    def synant_det():
        synonyms = []
        antonyms = []

        for syn in wordnet.synsets('good'):
            for l in syn.lemmas():
                print ('l:', l)
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
    
    # Comment in/out if needed:
    # synant_det()

    # Similarities - Symatics Similarities.
    # This method can help catch similar articles or plagarism.
    def sim_play():

        w1 = wordnet.synset("ship.n.01")
        w2 = wordnet.synset("boat.n.01")

        # Print the % similartity between 'w1' and 'w2'
        # 'wup' refers to Wu and Palmer - see research papers for more.
        print(w1.wup_similarity(w2))

        w3 = wordnet.synset("ship.n.01")
        w4 = wordnet.synset("car.n.01")
        print(w3.wup_similarity(w2))

        w5 = wordnet.synset("ship.n.01")
        w6 = wordnet.synset("cat.n.01")
        print(w5.wup_similarity(w6))

        w7 = wordnet.synset("ship.n.01")
        w8 = wordnet.synset("cactus.n.01")
        print(w7.wup_similarity(w8))

    # Comment in/out if needed:
    # sim_play()

    print('\n\n')

# Comment in/out to run this part:
part_10()


############### Part 11 ###############
# Text Classifying
# Sentiment Analysis
# In this case, just showing simple binary label.
# i.e. Positive or Negative.
# And not more or less positive/negative.

def part_11():
    import nltk
    import random
    from nltk.corpus import movie_reviews

    # List of tuples: [words, category]
    documents = [(list(movie_reviews.words(fileid)),category)
                  for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)] 
    
    # Randomise for training test.
    random.shuffle(documents)

    # print(documents[1]) # Comment in just to see.

    all_words = [] # may end up with a huge amount of words
    for w in movie_reviews.words():
        all_words.append(w.lower())
    
    all_words = nltk.FreqDist(all_words)
    print(all_words.most_common(15)) # top 15 most common 'words'.

    print (all_words["stupid"]) # How many times stupid comes up.

# Comment in/out to run this part:
# part_11()

############### Part 12 ###############
# Words as Features for Learning
# Continuation of Text Classification
# We have to find some way to "describe" bits of data, which are labeled as 
# either positive or negative for machine learning training purposes.

def part_12():
    import nltk
    import random
    from nltk.corpus import movie_reviews

    # List of tuples: [words, category]
    documents = [(list(movie_reviews.words(fileid)),category)
                  for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)] 
    
    # Randomise for training test.
    random.shuffle(documents)

    all_words = [] # may end up with a huge amount of words
    for w in movie_reviews.words():
        all_words.append(w.lower())
    
    all_words = nltk.FreqDist(all_words)

    # Up to the top 3k words, and only the keys.
    # This is used to train against.
    word_features = list(all_words.keys())[:3000]

    def find_features(document):
        # Converts a the first part of the tuple (which is a list of words in our case)
        # into a set, so we get only on iteration of any unique element. So we get all
        # the words and not just the amount of those words. Every single word will be 
        # included in the set of words.
        words = set(document) 

        features = {} # An empty dictionary

        for w in word_features:
            features[w] = (w in words) # A boolean value! If the word is in the 3k words.
        return features
    
    print ((find_features(movie_reviews.words('neg/cv000_29416.txt')))) # This is a negative review.


    # A list 
    # To classify and identify the words that appear commonly in a negative/positive review.
    featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Comment in/out to run this part:
# part_12()

############### Part 13 ###############
# Naive Bayes
# Classification - NB Algorithm 

def part_13():
    import nltk
    import random
    from nltk.corpus import movie_reviews

    # List of tuples: [words, category]
    documents = [(list(movie_reviews.words(fileid)),category)
                  for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)] 
    
    random.shuffle(documents)

    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())
    
    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:3000]

    def find_features(document):
        words = set(document) 
        features = {}

        for w in word_features:
            features[w] = (w in words)
        return features
    
    # print ((find_features(movie_reviews.words('neg/cv000_29416.txt')))) # This is a negative review.

    featuresets = [(find_features(rev), category) for (rev, category) in documents]

    training_set = featuresets[:1900]
    testing_set  = featuresets[1900:]

    # Bayes Algorithm - Scalable and Easy to understand.
    # Posterior = Prior Occurences * Likelihood / Current Evidence

    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print ('Naive Bayes Algo accuracy (%):', \
            nltk.classify.accuracy(classifier, testing_set)*100)

    classifier.show_most_informative_features(15)

# Comment in/out to run this part:
# part_13()


############### Part 14 ###############
# Save Classifier with Pickle 
# So you don't always have to run the classifier everytime.
# And mostly because the longest part to load is the documents.
#
# Pickle saves Python objects!

def part_14():
    import nltk
    import random
    from nltk.corpus import movie_reviews
    import pickle


    documents = [(list(movie_reviews.words(fileid)),category)
                  for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)] 
    
    random.shuffle(documents)

    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())
    
    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:3000]

    def find_features(document):
        words = set(document) 
        features = {}

        for w in word_features:
            features[w] = (w in words)
        return features

    featuresets = [(find_features(rev), category) for (rev, category) in documents]

    training_set = featuresets[:1900]
    testing_set  = featuresets[1900:]

    # classifier = nltk.NaiveBayesClassifier.train(training_set)

    # Loading the saved classifier:
    # (Make sure already saved first with 'save_class' below. If not yet, comment out)
    classifier_f = open('naivebayes.pickle', 'rb')
    classifier = pickle.load(classifier_f)

    print ('Naive Bayes Algo accuracy (%):', \
            nltk.classify.accuracy(classifier, testing_set)*100)

    classifier.show_most_informative_features(15)

    # Code to save the classifier:
    def save_class(classifier):
        save_classifier = open('naivebayes.pickle', 'wb') 
        pickle.dump(classifier, save_classifier)
        save_classifier.close()
    # Comment in/out if needed:
    # save_class(classifier)

# Comment in/out to run this part:
# part_14()


############### Part 15 ###############
# Scikit-Learn incorporation 
# Marrying the NLTK module with Scikit-Learn module.
# NLTK is not a machine learning toolkit, but Scikit-Learn is.
# 
# Make sure to install this first:
# pip install numpy
# pip install scipy
# pip install matplotlib
# pip install scikit-learn
# 

# def part_15():
#     import nltk
#     import random
#     from nltk.corpus import movie_reviews
#     from nltk.classify.scikitlearn import 
#     import pickle

# # Comment in/out to run this part:
# # part_15()

 
