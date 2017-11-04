# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random
import random

import math

import json
# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import logging
import sys

from create_data import *

DEBUG = False

# log = logging.getLogger()
# log.setLevel(logging.INFO)

# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# log.addHandler(ch)

#---------------------------------------------------------------------------------

def preprocess_text(input_negotiation):
    ''' Removes all punctuations except space and lowers all the letters'''

    line=remPunct(input_negotiation)

    new_line = ''

    for word in line:
        word = word.lower()
        preprocessed_line = new_line+word

    return preprocessed_line

#-------------------------------------------------------------------------
def get_gt_labels(input_json_file):
	gt_labels=[]

	with open(input_json_file) as json_data:
		data = json.load(json_data)
		data=data["all_reviews"]

		for i,val in enumerate(data):
			label=data[i]["sentiment"]
			gt_labels.append(label)

	return numpy.array(gt_labels)
#---------------------------------------------------------------------------
def predict_negotiation(input_json_file,doc2vec_model):

	all_labels=get_gt_labels(input_json_file)
	total_negotiation=len(all_labels)

	num_train_examples=int(math.ceil(total_negotiation/2))

	num_test_examples=int(math.floor(total_negotiation/2))

	model = Doc2Vec.load(doc2vec_model)
	#log.info('Negotiation')
	train_arrays = numpy.zeros((num_train_examples, 100))
	train_labels = numpy.zeros(num_train_examples)

	for i in range(num_train_examples):
		prefix_train_data = 'data_' + str(i)
		train_arrays[i] = model.docvecs[prefix_train_data]
		train_labels[i] = all_labels[i]


	#log.info(train_labels)

	test_arrays = numpy.zeros((num_test_examples, 100))
	test_labels = numpy.zeros(num_test_examples)

	for i in range(num_test_examples):
		prefix_test_data = 'data_' + str(num_train_examples+i)
		test_arrays[i] = model.docvecs[prefix_test_data]
		test_labels[i] = all_labels[num_train_examples+i]


	#log.info('Fitting')
	classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
	classifier.fit(train_arrays, train_labels)

	#print 'Negotiation Result'
	#log.info(classifier.score(test_arrays, test_labels))
	#log.info(classifier.predict(test_arrays))


#---------------------------------------------------------------------------

def predict_negotiation_test(input_json_file,doc2vec_model,test_negotiation):

	all_labels=get_gt_labels(input_json_file)

	total_negotiation=len(all_labels)

	model = Doc2Vec.load(doc2vec_model)
	#log.info('Negotiation')
	train_arrays = numpy.zeros((total_negotiation, 100))
	train_labels = numpy.zeros(total_negotiation)

	for i in range(total_negotiation):
		prefix_train_data = 'data_' + str(i)
		train_arrays[i] = model.docvecs[prefix_train_data]
		train_labels[i] = all_labels[i]


	# Infering the vector for test vector
	test_negotiation=preprocess_text(test_negotiation)
	test_negotiation=test_negotiation.split()

	#Parameters
	alpha=0.1
	min_alpha=0.0001
	steps=5

	d_test = model.infer_vector(doc_words=test_negotiation, alpha=alpha, min_alpha=min_alpha, steps=steps)

	test_arrays=numpy.zeros((1,100))

	for i in range(test_arrays.shape[0]):
		test_arrays[i]=d_test

	classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
	classifier.fit(train_arrays, train_labels)

	print 'Predict Movie Review Sentiment'
	print classifier.predict_proba(test_arrays)
	#print classifier.predict(test_arrays)
	#print classifier.predict_log_proba(test_arrays)

	#log.info(classifier.predict(test_arrays))
