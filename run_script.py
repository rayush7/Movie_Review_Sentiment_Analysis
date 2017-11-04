from create_data import *
from train_doc2vec import *
from predict_negotiation import *

# Extracts the Data from deal.json and stores into train_text_file.txt
extract_text_from_json('movie_review_data.json','movie_review_text_data.txt')

# train the doc2vec model on train_text_file
train_doc2vec('movie_review_text_data.txt','movie_review.d2v')

# predict the negotiation output
#predict_negotiation('movie_review_data.json','./movie_review.d2v')

#Predict the test negotiation
predict_negotiation_test('./movie_review_data.json','./movie_review.d2v','This movie is among the worst movies of this year. The direction is not good. A complete waste of money and time.')
