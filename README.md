# Movie_Review_Sentiment_Analysis
This repository contains code for movie review sentiment analysis as positive or negative.

# README #

### The repository has following dependencies:

* Anaconda for Python 2.7 (Linux Machine)
https://www.anaconda.com/download/#linux

* Gensim 2.3.0
https://radimrehurek.com/gensim/install.html
Note : The gensim library itself has many dependencies and they must be installed too. Please go to above link for details.

The code has been tested on Ubuntu 14.04 LTS.

### Labels:
* 1 : positive sentiment
* 0 : negative sentiment

### To run the script:
* python run_script.py

After execution training data file train_text_data.txt and Doc2Vec model movie_review.d2v will be generated. Further RandomForest Classifier has been used for classification of negotiations.

