#!/usr/bin/env python
#
# A modified version of libSVM's grid.py script. The original can be found in the
# 'tools' directory in any distribution of libSVM:
# http://www.csie.ntu.edu.tw/~cjlin/libsvm/
#
# Reference:
# Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines.
# ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.
# Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm
#
# This modified version allows optimizing on different criteria than accuracy.
# Optimizing on accuracy could cause problems with unbalanced data, where the
# classifier would achieve higher scores by binning everything into the majority
# class. Using F-measures should counter this problem.
#
# This mod also has some REDUCED FUNCTIONALITY compared to the original.
# Specifically, it does not support SSH or Telnet workers, multiple local
# workers are also not supported as using more than one worker would invariably
# result in segfaults or illegal instruction errors. If you wish to use these
# functions, please use the original grid.py.
#
# Depends on confusionmatrix.py by Vincent Van Asch:
# http://www.clips.ua.ac.be/scripts/confusionmatrix
#
# Script modified by
# Frederik Vaassen <frederik.vaassen@ua.ac.be>
# CLiPS Research Center, Antwerp, Belgium
#

import os, sys, traceback
import getpass
from operator import itemgetter
from random import shuffle
from itertools import chain
from threading import Thread
from subprocess import *
from optparse import OptionParser
from confusionmatrix import ConfusionMatrix

if(sys.hexversion < 0x03000000):
	import Queue
else:
	import queue as Queue

def range_f(begin,end,step):
	# like range, but works on non-integer too
	seq = []
	while True:
		if step > 0 and begin > end: break
		if step < 0 and begin < end: break
		seq.append(begin)
		begin = begin + step
	return seq

def permute_sequence(seq):
	n = len(seq)
	if n <= 1: return seq

	mid = int(n/2)
	left = permute_sequence(seq[:mid])
	right = permute_sequence(seq[mid+1:])

	ret = [seq[mid]]
	while left or right:
		if left: ret.append(left.pop(0))
		if right: ret.append(right.pop(0))

	return ret

def redraw(db,best_param,tofile=False):
	if len(db) == 0: return
	begin_level = round(max(x[2] for x in db)) - 3
	step_size = 0.5

	best_log2c,best_log2g,best_rate = best_param

	if tofile:
		gnuplot.write(b"set term png transparent small linewidth 2 medium enhanced\n")
		gnuplot.write("set output \"{0}\"\n".format(png_filename.replace('\\','\\\\')).encode())
		#gnuplot.write(b"set term postscript color solid\n")
		#gnuplot.write("set output \"{0}.ps\"\n".format(dataset_title).encode().encode())
	elif is_win32:
		gnuplot.write(b"set term windows\n")
	else:
		gnuplot.write( b"set term x11\n")
	gnuplot.write(b"set xlabel \"log2(C)\"\n")
	gnuplot.write(b"set ylabel \"log2(gamma)\"\n")
	gnuplot.write("set xrange [{0}:{1}]\n".format(c_begin,c_end).encode())
	gnuplot.write("set yrange [{0}:{1}]\n".format(g_begin,g_end).encode())
	gnuplot.write(b"set contour\n")
	gnuplot.write("set cntrparam levels incremental {0},{1},100\n".format(begin_level,step_size).encode())
	gnuplot.write(b"unset surface\n")
	gnuplot.write(b"unset ztics\n")
	gnuplot.write(b"set view 0,0\n")
	gnuplot.write("set title \"{0}\"\n".format(dataset_title).encode())
	gnuplot.write(b"unset label\n")
	gnuplot.write("set label \"Best log2(C) = {0}  log2(gamma) = {1}  {2} = {3}%\" \
				  at screen 0.5,0.85 center\n". \
				  format(best_log2c, best_log2g, criterion, best_rate).encode())
	gnuplot.write("set label \"C = {0}  gamma = {1}\""
				  " at screen 0.5,0.8 center\n".format(2**best_log2c, 2**best_log2g).encode())
	gnuplot.write(b"set key at screen 0.9,0.9\n")
	gnuplot.write(b"splot \"-\" with lines\n")




	db.sort(key = lambda x:(x[0], -x[1]))

	prevc = db[0][0]
	for line in db:
		if prevc != line[0]:
			gnuplot.write(b"\n")
			prevc = line[0]
		gnuplot.write("{0[0]} {0[1]} {0[2]}\n".format(line).encode())
	gnuplot.write(b"e\n")
	gnuplot.write(b"\n") # force gnuplot back to prompt when term set failure
	gnuplot.flush()

def calculate_jobs():
	c_seq = permute_sequence(range_f(c_begin,c_end,c_step))
	g_seq = permute_sequence(range_f(g_begin,g_end,g_step))
	nr_c = float(len(c_seq))
	nr_g = float(len(g_seq))
	i = 0
	j = 0
	jobs = []

	while i < nr_c or j < nr_g:
		if i/nr_c < j/nr_g:
			# increase C resolution
			line = []
			for k in range(0,j):
				line.append((c_seq[i],g_seq[k]))
			i = i + 1
			jobs.append(line)
		else:
			# increase g resolution
			line = []
			for k in range(0,i):
				line.append((c_seq[k],g_seq[j]))
			j = j + 1
			jobs.append(line)
	return jobs

def seed():
	return 0.7787265877145304

def split_problem(labels, features, num_folds):
	'''
	Returns a list of [num_folds] folds. Each fold is a dictionary containing a
	train and a test fold. Each train/test section is again a dictionary
	containing labels and features.

	To access, say, the training labels from the 4th fold, do:

	folds[3]['train']['labels']

	'''
	if num_folds < 2:
		sys.exit("Can't cross-validate on less than two folds!")
	else:
		shuffle(labels, seed)
		shuffle(features, seed)
		labels = [labels[i::num_folds] for i in range(num_folds)]
		features = [features[i::num_folds] for i in range(num_folds)]
		fold_labels = [{'train': list(chain.from_iterable(labels[:i]+labels[i+1:])), 'test': labels[i]} for i in range(num_folds)]
		fold_features = [{'train': list(chain.from_iterable(features[:i]+features[i+1:])), 'test': features[i]} for i in range(num_folds)]

		folds = []
		for fold_num in range(num_folds):
			train_fold = {'labels': fold_labels[fold_num]['train'], 'features': fold_features[fold_num]['train']}
			test_fold = {'labels': fold_labels[fold_num]['test'], 'features': fold_features[fold_num]['test']}
			folds.append({'train': train_fold, 'test': test_fold})

	return folds

class WorkerStopToken:  # used to notify the worker to stop
		pass

class Worker(Thread):
	def __init__(self,name,job_queue,result_queue):
		Thread.__init__(self)
		self.name = name
		self.job_queue = job_queue
		self.result_queue = result_queue
	def run(self):
		while True:
			(cexp,gexp) = self.job_queue.get()
			if cexp is WorkerStopToken:
				self.job_queue.put((cexp,gexp))
				break
			try:
				rate = self.run_one(2.0**cexp,2.0**gexp)
				if rate is None: raise RuntimeError("get no rate")
			except:
				# we failed, let others do that and we just quit
				traceback.print_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])

				self.job_queue.put((cexp,gexp))
				print('worker {0} quit.'.format(self.name))
				break
			else:
				self.result_queue.put((self.name,cexp,gexp,rate))

class LocalWorker(Worker):
	def run_one(self,c,g):
		# Read the problem.
		labels, features = svm_read_problem(dataset_pathname)

		# Split the problem into [fold] folds
		folds = split_problem(labels, features, fold)

		# Classify each fold
		paramstring = '-c {c} -g {g} -q {pts}'.format \
		  (c=c,g=g,pts=pass_through_string)
		cm = ConfusionMatrix()
		for fold_num in range(fold):
			train_labels = folds[fold_num]['train']['labels']
			train_features = folds[fold_num]['train']['features']
			test_labels = folds[fold_num]['test']['labels']
			test_features = folds[fold_num]['test']['features']
			# Train
			model = svm_train(train_labels, train_features, paramstring)
			# Test
			pred_labels, (ACC, MSC, SCC), pred_values = svm_predict(test_labels, test_features, model)
			assert len(test_labels) == len(pred_labels)
			for ref, pred in zip(test_labels, pred_labels):
				cm.update(ref, pred)

		if criterion == 'minorityboost':
			label_counts = dict([(label, cm.count(label)) for label in set(labels)])
			sorted_classes = sorted(label_counts.iteritems(), key=itemgetter(1))
			minority_class = sorted_classes[0][0]
			majority_class = sorted_classes[-1][0]

			rate = cm.microprecision()*100
			if cm.precision(majority_class) == 0.0 or cm.precision(minority_class) == 0.0:
				rate = 0.0
		elif criterion == 'macrofmeasure':
			rate = cm.macrofmeasure()*100
		elif criterion == 'microfmeasure':
			rate = cm.microfmeasure()*100
		elif criterion == 'accuracy':
			rate = cm.accuracy*100
		else:
			sys.exit('Unknown criterion: {0}\n Available: macrofmeasure, microfmeasure, accuracy, minorityboost'.format(criterion))

		return rate

def main():

	# put jobs in queue
	jobs = calculate_jobs()
	job_queue = Queue.Queue(0)
	result_queue = Queue.Queue(0)

	for line in jobs:
		for (c,g) in line:
			job_queue.put((c,g))

	# hack the queue to become a stack --
	# this is important when some thread
	# failed and re-put a job. It we still
	# use FIFO, the job will be put
	# into the end of the queue, and the graph
	# will only be updated in the end
	job_queue._put = job_queue.queue.appendleft

	# fire local worker
	LocalWorker('local',job_queue,result_queue).start()

	# gather results
	done_jobs = {}

	result_file = open(out_filename, 'w')

	db = []
	best_rate = -1
	best_c1,best_g1 = None,None

	for line in jobs:
		for (c,g) in line:
			while (c, g) not in done_jobs:
				(worker,c1,g1,rate) = result_queue.get()
				done_jobs[(c1,g1)] = rate
				result_file.write('{0} {1} {2}\n'.format(c1,g1,rate))
				result_file.flush()
				if (rate > best_rate) or (rate==best_rate and g1==best_g1 and c1<best_c1):
					best_rate = rate
					best_c1,best_g1=c1,g1
					best_c = 2.0**c1
					best_g = 2.0**g1
				print("[{0}] {1} {2} {3} (best c={4}, g={5}, rate={6})".format \
			(worker,c1,g1,rate, best_c, best_g, best_rate))
			db.append((c,g,done_jobs[(c,g)]))
		redraw(db,[best_c1, best_g1, best_rate])
		redraw(db,[best_c1, best_g1, best_rate],True)

	job_queue.put((WorkerStopToken,None))
	print("{0} {1} {2}".format(best_c, best_g, best_rate))

if __name__ == '__main__':

	parser = OptionParser(usage = "python %prog data_file (options)\nOptimizes libSVM's c an g parameters on data_file.", version='%prog 0.1')
	parser.add_option('-c', '--log2c', dest='log2c', default=None,
						help='log2 of start, end and step-values (comma-separated) to use to search the c parameter. (Default: -5,15,2)')
	parser.add_option('-g', '--log2g', dest='log2g', default=None,
						help='log2 of start, end and step-values (comma-separated) to use to search the g parameter. (Default: 3,-15,-2)')
	parser.add_option('-v', '--cv-folds', dest='folds', default=5, type='int',
						help="Specify the number of folds to be used for cross-validation. (Default: 5)")
	parser.add_option('-r', '--criterion', dest='criterion', default='macrofmeasure',
						help="Specify the criterion to use for optimization. (Allowed: accuracy, microfmeasure, macrofmeasure (default)")
	parser.add_option('-o', '--out-file', dest='out_filename', default=None,
						help="Choose the location of the output file. (Default: data_file.out)")
	parser.add_option('-p', '--png-file', dest='png_filename', default=None,
						help="Choose the location of the PNG file that will contain the plot. (Default: data_file.png)")
	parser.add_option('--libsvm-path', dest='libsvmpath', default='/opt/libsvm',
						help="Specify the path to the libSVM folder. (Default: /opt/libsvm)")
	parser.add_option('--gnuplot-path', dest='gnuplot_exe', default='/usr/bin/gnuplot',
						help="Specify the location of Gnuplot. (Default: /usr/bin/gnuplot)")
	parser.add_option('--passthrough', dest='passthrough', default='',
						help="Passthrough string for the classifier. (Default: "")")
	(options, args) = parser.parse_args()

	if len(args) < 1:
		sys.exit(parser.print_help())

	# Input file
	dataset_pathname = args[0]

	is_win32 = (sys.platform == 'win32')
	if not is_win32:
		svmtrain_exe = os.path.join(options.libsvmpath, 'svm-train')
		gnuplot_exe = options.gnuplot_exe
	else:
		# example for windows
		svmtrain_exe = os.path.join(libsvmpath, 'svm-train.exe')
		gnuplot_exe = options.gnuplot_exe

	assert os.path.exists(svmtrain_exe),"svm-train executable not found"
	assert os.path.exists(gnuplot_exe),"gnuplot executable not found"
	assert os.path.exists(dataset_pathname),"dataset not found"

	sys.path.insert(0, os.path.join(options.libsvmpath, 'python'))
	from svmutil import *

	# Output files
	dataset_title = os.path.basename(dataset_pathname)
	if not options.out_filename:
		out_filename = '{0}.out'.format(dataset_title)
	else:
		out_filename = options.out_filename
	if not options.png_filename:
		png_filename = '{0}.png'.format(dataset_title)
	else:
		png_filename = options.png_filename

	pass_through_string = options.passthrough

	fold = options.folds
	criterion = options.criterion

	if not options.log2c:
		c_begin, c_end, c_step = -5,  15, 2
	else:
		c_begin, c_end, c_step = map(float, options.log2c.split(','))
	if not options.log2g:
		g_begin, g_end, g_step =  3, -15, -2
	else:
		g_begin, g_end, g_step = map(float, options.log2g.split(','))

	gnuplot = Popen(gnuplot_exe,stdin = PIPE).stdin
	main()
