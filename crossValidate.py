#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#	  crossValidate.py
#
#	  Frederik Vaassen <frederik.vaassen@ua.ac.be>
#	  Copyright 2012 CLiPS Research Center
#
#	  This program is free software; you can redistribute it and/or modify
#	  it under the terms of the GNU General Public License as published by
#	  the Free Software Foundation; either version 2 of the License, or
#	  (at your option) any later version.
#
#	  This program is distributed in the hope that it will be useful,
#	  but WITHOUT ANY WARRANTY; without even the implied warranty of
#	  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	  GNU General Public License for more details.
#
#	  You should have received a copy of the GNU General Public License
#	  along with this program; if not, write to the Free Software
#	  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#	  MA 02110-1301, USA.
#

import re
import os
import sys
import math
import time
import operator
from collections import defaultdict
from optparse import OptionParser
from xml.etree.ElementTree import ElementTree
from confusionmatrix import ConfusionMatrix

liblinearpath = '/home/frederik/Tools/liblinear-1.8'
libsvmpath = '/home/frederik/Tools/libsvm-3.11'

classifier = 'libsvm'

def getLabelMap(folder):
	'''
	Retrieves the label map from the first svm.metadata.txt in <folder>.

	'''
	for root, dirs, files in os.walk(os.path.abspath(folder)):
		if 'svm.metadata.txt' in files:
			with open(os.path.join(root, 'svm.metadata.txt'), 'r') as fin:
				lines = [line.strip().split(' > ') for line in fin.readlines()]
				mapping = {}
				for line in lines:
					mapping[line[1].split('Class_')[1]] = int(line[0])
				return mapping

def invertLabelMap(label_map):
	return dict([(value, key) for key, value in label_map.items()])

def getInstances(folder, pattern=None):
	'''
	Returns a list of tuples containing the instance files for (train, test) per
	fold.
	Loops through <folder> and return all SVM instance files matching the
	specified filters. If no filter is specified, try to find the unfiltered
	token unigram TotalSet files. Falls back to the first instance file in
	alphabetical order.

	Output:
	[(fold-01.train, fold-01.test), (fold-02.train, fold-02.test),...]

	'''
	folds = sorted([os.path.join(os.path.abspath(folder), f) for f in os.listdir(folder) if re.match('fold-\d+', f)])

	if pattern:
		pattern = re.compile(pattern)
	else:
		pattern = re.compile('N_GRAM_TOKEN_?1.TotalSet.ngrams.txt')

	instances = []
	for fold in folds:
		fold_inst = [None, None]
		for root, dirs, files in os.walk(fold):
			path = os.path.split(root)
			if path[1] == 'svm':
				path = os.path.split(path[0])
				if path[1] == 'train':
					fold_inst[0] = sorted([os.path.join(root, f) for f in files if re.search(pattern, f)])[0]
				elif path[1] == 'test':
					fold_inst[1] = (sorted([os.path.join(root, f) for f in files if re.search(pattern, f)])[0] or None)
		assert fold_inst[0] is not None
		assert fold_inst[1] is not None
		instances.append(tuple(fold_inst))

	return instances

def getRemapping(xml_file):
	'''
	Read the XML file containing label remappings.

	'''
	tree = ElementTree()
	tree.parse(xml_file)
	remapping = dict([(m.attrib['source'], m.attrib['dest']) for m in tree.findall('remap')])

	return remapping

def reMap(labels, mapping):
	'''
	If a label mapping is provided, change the labels according to this mapping.

	'''
	if mapping:
		return [int(mapping[label]) for label in labels]
	else:
		return labels

def train(train_file, fold_num, mapping=None, parameters={'-c': 1}, output_folder=None):
	'''
	Given a training instance file and (optionally) a label mapping, adapt the
	training vectors to fit the mapping and build an SVM model.

	'''
	global classifier
	if not output_folder:
		output_folder = 'models'
	output_folder = os.path.join(output_folder, 'fold-{0:02d}'.format(fold_num+1))
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	labels, instances = svm_read_problem(train_file)
	labels = reMap(labels, mapping)

	# Exclude instances which have 0 as their label
	labels, instances = zip(*[(label, instance) for label, instance in zip(labels, instances) if label != 0])

	distribution = {}
	for label in set(labels):
		distribution[label] = float(labels.count(label))/len(labels)

	paramstring = ''
	for param, value in parameters.items():
		paramstring += ' {0} {1}'.format(param, value)
	if classifier == 'libsvm' and '-b' not in parameters.keys():
		paramstring += ' -b 1'
	paramstring += ' -q'

	model_file = os.path.join(output_folder, os.path.basename(train_file) + '.model')
	print '---training'
	model = svm_train(labels, instances, paramstring)
	svm_save_model(model_file, model)

	return model_file, distribution

def test(test_file, model_file, fold_num, mapping=None, debug=False):
	'''
	Returns predicted labels, prediction values and the as the the test labels
	(potentially remapped).

	Requires a test instance file and a corresponding model
	file. Remaps the labels in the test file (optional), classifies the test
	instances against the model.

	'''
	labels, instances = svm_read_problem(test_file)
	labels = reMap(labels, mapping)

	# Exclude instances which have 0 as their label
	labels, instances = zip(*[(label, instance) for label, instance in zip(labels, instances) if label != 0])

	if debug:
		with open(os.path.basename(test_file) + '.remap', 'w') as fout:
			for label, instance in zip(labels, instances):
				output = '{0} '.format(str(label))
				for idx, val in instance.items():
					output += '{0}:{1} '.format(str(idx), str(val))
				output = output.strip() + '\n'
				fout.write(output)

	model = svm_load_model(model_file)

	print '---testing'
	if classifier == 'liblinear':
		pred_labels, ACC, pred_values, label_order = svm_predict(labels, instances, model)
	elif classifier == 'libsvm':
		pred_labels, (ACC, MSC, SCC), pred_values = svm_predict(labels, instances, model, options='-b 1')
		label_order = model.get_labels()

	return pred_labels, pred_values, label_order, labels

def main(instance_folder, results_folder, remap_file=None, parameters={'-c': 1}, pattern=None):
	'''
	Returns a list of lists of predicted labels (per fold).

	Contains the main pipeline:

	* Remaps the class labels if a mapping is specified.

	* For each fold, trains and tests each fold based on the instances in
	<instance_folder>. By default the script looks for files with the following
	filenames:
	N_GRAM_TOKEN_?1.TotalSet.ngrams.txt or
	Use the <pattern> option to specify your own pattern (can be a regex).

	'''
	instance_files = getInstances(instance_folder, pattern=pattern)
	labels = getLabelMap(instance_folder)

	'''
	Get all class (re-)mappings.

	Example:
	original label	original SVM	remapped	remapped SVM
	A				1				X			1
	B				2				Y			2
	C				3				Y			2
	D				4				Y			2

	The stylene_mapping maps from the original label to its SVM equivalent
	The intermediate mapping maps from "orginal SVM" to "remapped SVM"
	The true_mapping maps from "remapped" to its SVM equivalent

	'''
	true_mapping = {}
	stylene_mapping = getLabelMap(instance_folder)
	if remap_file:
		intermediate_mapping = {}
		remapping = getRemapping(remap_file)
		assert not False in [(l in remapping) for l in stylene_mapping], "Your label remapping file doesn't contain all classes found in the Stylene mapping!"
		for i, label in enumerate(sorted(set(remapping.values()))):
			# Set instances to be excluded.
			if label == 'EXCLUDE':
				true_mapping[label] = 0
			else:
				true_mapping[label] = i+1

		for label, mapped in stylene_mapping.items():
			intermediate_mapping[mapped] = true_mapping[remapping[label]]
	else:
		intermediate_mapping = None
		true_mapping = stylene_mapping

	# Invert for reverse lookup
	true_mapping = invertLabelMap(true_mapping)

	gold_labels = []
	predicted_labels = []
	predicted_values = defaultdict(list)
	global_cm = ConfusionMatrix()
	fold_accuracies = []
	# Train and test each fold.
	for i, (train_file, test_file) in enumerate(instance_files):

		# If the dictionary file exists, use it to add more meta-information.
		base_name = test_file.split('.ngrams.txt')[0]
		dict_file = base_name + '.dictionary.ngrams.txt'
		base_name = os.path.basename(base_name.split('.TotalSet')[0])
		if not os.path.exists(dict_file):
			original_filenames = None
		else:
			original_filenames = []
			with open(dict_file, 'r') as fin:
				while True:
					line = fin.readline()
					if not line.strip():
						break
					original_filenames.append(line.split()[1])

		print 'Processing fold {0}/{1}'.format(i+1, len(instance_files))
		# Train!
		fold_model, fold_distribution = train(train_file, i, intermediate_mapping, parameters, results_folder)
		print 'Training distribution:', dict([(true_mapping[label], dist) for label, dist in fold_distribution.items()])

		# Classify!
		pred_labels, pred_values, label_order, test_labels = test(test_file, fold_model, i, intermediate_mapping)

		# Write the predicted (non-remapped) labels to file.
		pred_values_dict = dict([(label, []) for label in label_order])
		output_folder = os.path.join(results_folder, 'fold-{0:02d}'.format(i+1))

		with open(os.path.join(output_folder, os.path.basename(test_file) + '.predicted'), 'w') as fout:
			# Headers
			fout.write('filename predicted\t')
			for label in label_order:
				fout.write('{0}\t'.format(label))
			fout.write('{0}\n'.format(dict([(label, true_mapping[label]) for label in label_order])))
			# Predictions
			for j, (pred_label, pred_vals) in enumerate(zip(pred_labels, pred_values)):
				for k, value in enumerate(pred_vals):
					pred_values_dict[label_order[k]].append(value)
				if not original_filenames:
					filename = 'unknown'
				else:
					filename = original_filenames[j].split('.ProcessedFrequencies.xml')[0].split(base_name)[0].rstrip('.')
				fout.write('{0}\t{1}\t{2}\n'.format(filename, int(pred_label), ' '.join([str(val) for val in pred_vals])))

		# Remap to the true labels
		pred_labels = [true_mapping[label] for label in pred_labels]
		predicted_labels.extend(pred_labels)
		g_labels = [true_mapping[label] for label in test_labels]
		gold_labels.extend(g_labels)

		local_cm = ConfusionMatrix()
		for (gold, pred) in zip(g_labels, pred_labels):
			local_cm.update(gold, pred)
			global_cm.update(gold, pred)

		# Update the predicted values
		pred_values = dict([(true_mapping[label], values) for label, values in pred_values_dict.items()])
		assert i==0 or sorted(pred_values) == sorted(predicted_values)
		for label, values in pred_values.items():
			predicted_values[label].extend(values)

		print local_cm
		print 'Fold Accuracy: {0:.3f}'.format(local_cm.accuracy)
		fold_accuracies.append(local_cm.accuracy)

	avg_acc = sum(fold_accuracies)/len(fold_accuracies)
	stdev = (sum([(acc - avg_acc)**2 for acc in fold_accuracies])/len(fold_accuracies))**0.5
	print
	print 'Global confusion matrix:'
	print global_cm
	print
	for label in global_cm.classes:
		print '{0} P\t{1:.3f}'.format(label, global_cm.precision(label))
		print '{0} R\t{1:.3f}'.format(label, global_cm.recall(label))
		print '{0} F\t{1:.3f}'.format(label, global_cm.fmeasure(label))
	print
	print 'Accuracy: {0:.3f} +/- {1:.3f} ({2}/{3})'.format(global_cm.accuracy, stdev, sum([global_cm.TP(c) for c in global_cm.classes]), len(global_cm))
	print 'Micro-avg F:', global_cm.microfmeasure()
	print 'Macro-avg F:', global_cm.macrofmeasure()

if __name__ == '__main__':

	parser = OptionParser(usage = '''
python %prog instance_folder (options)

Takes a folder of Stylene train and test instance files (as structured by
styleneFolding.py) as input. Trains and tests each fold, remapping classes
if specified.''', version='%prog 0.1')
	parser.add_option('-i', '--instance-pattern', dest='pattern', default=None,
						help='Specify the file name of the instance files you want to use, can be a regex pattern. (Default: N_GRAM_TOKEN_?1.TotalSet.ngrams.txt)')
	parser.add_option('-o', '--output-folder', dest='output_folder', default=None,
						help="Specify the folder you want model files and predictions to be stored. (Default: instance_folder/fold-XX/)")
	parser.add_option('-c', '--classifier', dest='classifier', default='libsvm',
						help="Specify the classifier you want to use. (Available: 'libsvm' (default), 'liblinear')")
	parser.add_option('--params', dest='parameters', default="c:32-g:8",
						help="Specify the parameters for the classifier. Separate keys and values with a colon, and different parameters with a dash. (Default: 'c:512-g:8')")
	parser.add_option('-r', '--remap-file', dest='remap_file', default=None,
						help="Specify the location of the XML file containing a class remapping. (Default: None)")
	(options, args) = parser.parse_args()

	if len(args) != 1:
		sys.exit(parser.print_help())

	input_folder = args[0]
	if options.output_folder is None:
		output_folder = input_folder
	else:
		output_folder = options.output_folder

	if options.classifier == 'libsvm':
		sys.path.insert(0, os.path.join(libsvmpath, 'python'))
		from svmutil import svm_read_problem
		from svmutil import svm_train
		from svmutil import svm_save_model
		from svmutil import svm_load_model
		from svmutil import svm_predict
	elif options.classifier == 'liblinear':
		sys.path.insert(0, os.path.join(liblinearpath, 'python'))
		from liblinearutil import read_problem as svm_read_problem
		from liblinearutil import train as svm_train
		from liblinearutil import save_model as svm_save_model
		from liblinearutil import load_model as svm_load_model
		from liblinearutil import predict as svm_predict
	else:
		sys.exit('Unknown classifier "{0}".\nAllowed: "liblinear", "libsvm".'.format(options.classifier))
	classifier = options.classifier

	params = dict([('-'+p.split(':')[0], p.split(':')[1]) for p in options.parameters.strip().split('-')])

	main(input_folder, output_folder, remap_file=options.remap_file, parameters=params, pattern=options.pattern)
