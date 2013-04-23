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
from random import sample
from itertools import chain
from threading import Thread
from subprocess import *
from optparse import OptionParser
from confusionmatrix import *

if not __name__ == '__main__':

	quiet = True
	gnuplot_exe = '/usr/bin/gnuplot'
	libsvmpath = '/home/frederik/Tools/libsvm-3.16'
	sys.path.insert(0, os.path.join(libsvmpath, 'python'))
	from svmutil import *

if(sys.hexversion < 0x03000000):
	import Queue
else:
	import queue as Queue
is_win32 = (sys.platform == 'win32')


def svm_read_problem(data_file_name):
	"""
	svm_read_problem(data_file_name) -> [y, x]

	Read LIBSVM-format data from data_file_name and return labels y
	and data instances x.
	"""
	# From libSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/), amended
	# to allow for comments at the end of instances (preceded by #) and
	# to allow for multi-label instances (labels separated by ',').
	prob_y = []
	prob_x = []
	comments = []
	for line in open(data_file_name):
		line = line.split(None, 1)
		# In case an instance with all zero features
		if len(line) == 1: line += ['']
		labels, features = line
		tmp = features.split('#')
		if len(tmp) == 2:
			features = tmp[0].strip()
			comment = tmp[1].strip()
		else:
			comment = None
		xi = {}
		for e in features.split():
			ind, val = e.split(":")
			xi[int(ind)] = float(val)
		prob_y += [(map(float, labels.split(',')))]
		prob_x += [xi]
		comments += [comment]

	return (prob_y, prob_x), comments

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

def stratifiedSample(instances, ratio=0.1):

	(labels, features), comments = instances

	print len(labels)
	assert len(labels) >= 1, 'No instances!'
	sample_size = int(ratio*len(labels)+0.5)
	assert sample_size >= 100, 'Only {0} instances in sample!'.format(sample_size)

	print '------- Resampling using a ratio of {0} (using {1} instances)'.format(ratio, sample_size)

	label_set = set(labels)
	instances = zip(labels, features, comments)
	resampled_instances = []
	for label in label_set:
		label_instances = [(l, feats, comment) for (l, feats, comment) in instances if l==label]
		resampled_instances.extend(sample(label_instances, int(ratio*len(label_instances)+0.5)))

	labels, features, comments = zip(*resampled_instances)

	return (list(labels), list(features)), list(comments)

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
		if __name__ == '__main__':
			# Read the problem.
			(labels, features), comments = svm_read_problem(dataset_pathname)
			if devset_pathname:
				(dev_labels, dev_features), dev_comments = svm_read_problem(devset_pathname)
			else:
				# Split the problem into [fold] folds
				folds = split_problem(labels, features, fold)
		else:
			(labels, features), comments = train_instances
			if dev_instances:
				(dev_labels, dev_features), dev_comments = dev_instances
			else:
				# Split the problem into [fold] folds
				folds = split_problem(labels, features, fold)

		# Classify each fold
		paramstring = '-c {c} -g {g} -q'.format \
		  (c=c,g=g)
		cm = ConfusionMatrix()
		if fold:
			for fold_num in range(fold):
				train_labels = folds[fold_num]['train']['labels']
				train_features = folds[fold_num]['train']['features']
				test_labels = folds[fold_num]['test']['labels']
				test_features = folds[fold_num]['test']['features']
				# Train
				model = svm_train(train_labels, train_features, paramstring)
				# Test
				pred_labels, (ACC, MSC, SCC), pred_values = svm_predict(test_labels, test_features, model, '-q')
				assert len(test_labels) == len(pred_labels)
				for ref, pred in zip(test_labels, pred_labels):
					cm.single_add(ref, pred)
		else:
			# Train
			model = svm_train(labels, features, paramstring)
			# Test
			pred_labels, (ACC, MSC, SCC), pred_values = svm_predict(dev_labels, dev_features, model)
			assert len(dev_labels) == len(pred_labels)
			for ref, pred in zip(dev_labels, pred_labels):
				cm.single_add(ref, pred)
			# Setting labels for minority class calculations
			labels = dev_labels

		if criterion == 'minorityboost':
			label_counts = dict([(label, cm.count(label)) for label in set(labels)])
			sorted_classes = sorted(label_counts.iteritems(), key=itemgetter(1))
			minority_class = sorted_classes[0][0]
			majority_class = sorted_classes[-1][0]

			rate = cm.fscoree(minority_class)*100
			if cm.fscore(majority_class) == 0.0:
				rate = 0.0
		elif criterion == 'macrofmeasure':
			rate = cm.averaged(level=MACRO, score=FSCORE)*100
		elif criterion == 'microfmeasure':
			rate = cm.averaged(level=MICRO, score=FSCORE)*100
		elif criterion == 'accuracy':
			rate = cm.accuracy()*100
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
				if not quiet:
					print("[{0}] {1} {2} {3} (best c={4}, g={5}, rate={6})".format \
			(worker,2.0**c1,2.0**g1,rate, best_c, best_g, best_rate))
			db.append((c,g,done_jobs[(c,g)]))
		redraw(db,[best_c1, best_g1, best_rate])
		redraw(db,[best_c1, best_g1, best_rate],True)

	job_queue.put((WorkerStopToken,None))
	if not quiet:
		print("Best results: -c {0} -g {1} ({2})".format(best_c, best_g, best_rate))
	return best_c, best_g, best_rate

def gridSearch(train_inst, dev_inst=None, log2c=(-5, 15, 2), log2g=(3, -15, -2), folds=5, resample_ratio=None, crit='macrofmeasure', out_file='grid.out', img_file='grid.png'):

	# Initialize global variables, because grid.py is badly made.
	global dataset_title
	dataset_title = 'External dataset'
	global train_instances
	global dev_instances
	global c_begin, c_end, c_step
	global g_begin, g_end, g_step
	global fold
	global criterion
	global out_filename
	global png_filename
	train_instances = train_inst
	dev_instances = dev_inst
	c_begin, c_end, c_step = log2c
	g_begin, g_end, g_step = log2g
	fold = folds
	criterion = crit
	out_filename = out_file
	png_filename = img_file

	if resample_ratio:
		train_instances = stratifiedSample(train_instances, resample_ratio)

	global gnuplot
	gnuplot = Popen(gnuplot_exe,stdin = PIPE).stdin
	best_c, best_g, best_rate = main()

	#~ return ' -c {0} -g {0}'.format(best_c, best_g)
	return best_c, best_g, best_rate

if __name__ == '__main__':

	parser = OptionParser(usage = '''python %prog data_file [dev_file] (options)
Optimizes libSVM's c an g parameters on data_file.
If dev_file is present, instead of cross-validating, the script will
train on data_file and test on dev_file.''', version='%prog 0.1')
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
	parser.add_option('-q', '--quiet', dest='quiet', default=False, action="store_true",
						help="Don't print output for each iteration.")
	parser.add_option('--libsvm-path', dest='libsvmpath', default='/opt/libsvm',
						help="Specify the path to the libSVM folder. (Default: /opt/libsvm)")
	parser.add_option('--gnuplot-path', dest='gnuplot_exe', default='/usr/bin/gnuplot',
						help="Specify the location of Gnuplot. (Default: /usr/bin/gnuplot)")
	(options, args) = parser.parse_args()

	if len(args) not in [1,2]:
		sys.exit(parser.print_help())

	# Input file
	dataset_pathname = args[0]
	if len(args) == 2:
		devset_pathname = args[1]
	else:
		devset_pathname = None

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
	assert os.path.exists(dataset_pathname),"dataset not found:\n{0}".format(dataset_pathname)
	if devset_pathname:
		assert os.path.exists(devset_pathname),"devset not found:\n{0}".format(devset_pathname)

	sys.path.insert(0, os.path.join(options.libsvmpath, 'python'))
	from svmutil import *

	# Output files
	if devset_pathname:
		dataset_title = os.path.basename(devset_pathname)
	else:
		dataset_title = os.path.basename(dataset_pathname)
	if not options.out_filename:
		out_filename = '{0}.out'.format(dataset_title)
	else:
		out_filename = options.out_filename
	if not options.png_filename:
		png_filename = '{0}.png'.format(dataset_title)
	else:
		png_filename = options.png_filename

	if devset_pathname:
		print 'Dev set found, ignoring "folds" parameter.'
		fold = None
	else:
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

	quiet = options.quiet

	gnuplot = Popen(gnuplot_exe,stdin = PIPE).stdin
	main()
