import json
import re

#--------------------------------------------------
def preprocess_text(input_negotiation):
    ''' Removes all punctuations except space and lowers all the letters'''
    line=remPunct(input_negotiation)
    for word in line:
        word = word.lower()
    print line

#--------------------------------------------------
def remPunct(sentence):
    """
    getSentence("Example paragraph. of words, 'with randomness?")
    Example paragraph of words with randomness
    """
    return re.sub(r'([^\s\w]|_)+', '', sentence)

#---------------------------------------------------
def get_gt_labels(input_json_file):
	gt_labels=[]

	with open(input_json_file) as json_data:
		data = json.load(json_data)
		data=data["all_reviews"]

		for i,val in enumerate(data):
			label=data[i]["sentiment"]
			gt_labels.append(label)

	return gt_labels

#-----------------------------------------------------------------------------
def extract_text_from_json(input_json_file,output_text_file):

	with open(input_json_file) as json_data:

		text_file=open(output_text_file,'w')

		data = json.load(json_data)
		data=data["all_reviews"]

		for i,val in enumerate(data):
			line=str(data[i]["review"])
			line=remPunct(line)

			for word in line:
				text_file.write(word.lower())
			text_file.write('\n')

		text_file.close()
