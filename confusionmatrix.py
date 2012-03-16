#!/usr/bin/env python
'''Compute a confusion matrix.

Can be used to produce the metrics you get with the Timbl option +v cm+cs
from the Timbloutput file.

Example:

The Timbl commando "Timbl -f dimin.train -t dimin.test +v cs+cm"
gave an outputfile called : dimin.test.IB1.O.gr.k1.out
    
    >>> cm = getConfusionMatrixFromFile('dimin.test.IB1.O.gr.k1.out', FS=',')
    >>> print cm
                                predicted
          |   E |   J |   K |   P |   T
      -----------------------------------
    r   E |  87 |   4 |   8 |   1 |   0
    e   J |   4 | 347 |   0 |   0 |   1
    f   K |   7 |   0 |   9 |   0 |   0
        P |   3 |   0 |   0 |  24 |   0
        T |   0 |   2 |   0 |   0 | 453
    >>> print cm.macrofmeasure()
    0.863059988682
    >>> print cm.accuracy
    0.968421052632
    ...
    
For more info:
    >>> help(cm) 
    
Note: 
    The macro-averaged and micro-averaged f-scores may differ from the scores that are given
    by Timbl when using the +v cs option.
    
    Timbl uses the number of different classes in training for the macro-averaged scores.
    This script uses the number of different classes in test. Therefore, if a class occurs
    in training and not in test the macro-averaged scores of Timbl and this script will differ.
    A difference between Timbl and this script will not be observed very often since most of
    the time all classes of training are also in test and vice versa.
    
    Timbl uses the frequency of the classes in training when computing micro-averaged scores whereas
    this script uses the frequencies in test. Because the frequencies in training and test often differ
    the micro-averaged scores of Timbl and this script will also differ often.

# License: GNU General Public License, see http://www.clips.ua.ac.be/~vincent/scripts/LICENSE.txt
'''
__author__='Vincent Van Asch'
__date__ = 'March 2012'
__version__ = '1.2.1'

import os, tempfile, re, sys

class Error(Exception):
    def ___str__(self):
        return self.message
        
class FieldSeparatorError(Error):
    def __init__(self, fs):
        self.message='Connot find a given and predicted class. Possible FS is not set correctly as %s.' %str(fs)
        self.args=(self.message, fs)

class EmptyError(Error):
    def __init__(self):
        self.message='Confusionmatrix contains no values.'
        self.args=(self.message,)

class ClassError(Error):
    def __init(self, classe):
        self.message = '"%s" is not a known class of the matrix' %str(classe)
        self.args = (self.message, classe)
        
class TrainingError(Error):
    def __init__(self):
        self.message = 'confusionmatrix does not have information about a training file'
        self.args=(self.message,)

def reformat(instancefile, classfile, confusionmatrixfile=None, classindex=-1, FS=None, cutoff=None):
    '''Takes an instancefile and a file with predicted classes and appends the labels to the instance file.
    The outputfile confusionmatrixfile is suited to construct a confusionmatrix.
    Return the path of the confusionmatrixfile.
    
    Format of instancefile
        feat1 feat2 feat3 class
        feat1 feat2 feat3 class
        ...
    Format of classfile
        class
        class
        ...
        
    classindex: in the example above classindex is -1 which means the classlabel
    is the last string on every line. When the classlabel is the first on every line (e.g maxent) 
    you may specify classindex=0 
    FS: field separator in instancefile
    confusionmatrixfile: name of outputfile, if None stores the output in a tempfile
    cutoff: values in classfile below this cutoff get label -1; the others get 1. If None don't cut off.
    
    
    Note: if the class file contains more fields (separated by FS), it will only take into account the first field.
    '''
    t=open(os.path.expanduser(instancefile), 'rU')
    p=open(os.path.expanduser(classfile), 'rU')
    
    if confusionmatrixfile:
        output=os.path.expanduser(confusionmatrixfile)
    else:
        fd, output = tempfile.mkstemp()
        os.close(fd)
    
    o=open(output, 'w')
    
    try:
        for l in t:
            parts=l.strip().split(FS)
            pred=p.readline().strip().split(FS)[0]
            
            if cutoff is not None:
                try:
                    v = float(pred)
                except ValueError:
                    print >>sys.stderr, 'Error: can only use cutoff -C on numeric class labels'
                    sys.exit(1)
                    
                if v < cutoff:
                    pred = '-1'
                else:
                    pred = '1'
            
            ref = parts.pop(classindex)
            
            parts.append(ref)
            parts.append(pred)
            
            if FS is None:
                o.write(' '.join(parts)+'\n')
            else:
                o.write(FS.join(parts)+'\n')
    finally:
        t.close()
        p.close()
        o.close()
    
    return output
    
    

def _read(fname, FS):
    '''Reads in a file fname and returns an iterator. Where 
every yielded value is a list of the features in the instances.
The format of the file must be: an instance + gold + predicted on
a newline.
FS: the fieldseparator, if None splits on all whitespace'''
    f=open(os.path.abspath(os.path.expanduser(fname)), 'rU')
    
    try:
        for l in f:
            line=l.strip()
            if line and line[0] != '#':
                #Remove distribution
                line = re.sub('{[^{]+?}$', '', line)
                yield line.strip().split(FS)
    finally:
        f.close()

def getConfusionMatrixFromFile(fname, FS=None, training=None, translateclass=None, classindex=-1):
    """
    Returns an instance of a ConfusionMatrix for the file fname (path string).
    The format of a line the file must be:
            feature1FS...FSfeaturenFSgoldlabelFSpredictedlabel

    FS: (string) The field separator. If None, splits on all whitespace.
    training: (path string) The training file used to get the predictions in fname.
                            This is used to compute averaged scores like Timbl does.
              or  
              (Training)     A Training instance

    translateclass: (dictionary) If provided, all class labels are translated using this dict.
                                  
    classindex: (int) If training is a filename, this should be the index of the class label.
    """

    if training and not isinstance(training, Training):
        training = getTraining(training, classindex=classindex, FS=FS, translateclass=translateclass)

    cm = ConfusionMatrix(training=training)
    try:
        for line in _read(fname, FS=FS):
            gold = line[-2]
            pred = line[-1]
        
            if translateclass:
                try:
                    gold = translateclass[gold]
                except KeyError:
                    print >>sys.stderr, 'ERROR: old gold class label %s not present in translate file (option -T)' %(gold) 
                    raise
                try:
                    pred = translateclass[pred]
                except KeyError:
                    print >>sys.stderr, 'ERROR: old predicted class label %s not present in translate file (option -T)' %(pred) 
                    raise
            cm.update(gold, pred)
    except IndexError:
        raise FieldSeparatorError(FS)
    return cm


class Training(dict):
    def __init__(self, fname, d={}):
        dict.__init__(self, d)
        self._fname = fname
    @property
    def fname(self):
        '''The training filename'''
        return self._fname
class Translate(dict):
    def __init__(self, id, d={}):
        dict.__init__(self, d)
        self._fname = id
    @property
    def id(self):
        '''The id'''
        return self._fname

def getTraining(fname, classindex=-1, FS=None, translateclass=None):
    '''Returns a Training instance with the distribution of the 
    classes in the given file fname.
    
    classindex: (int) This should be the index of the class label.
    FS: (string) The field separator. If None, splits on all whitespace.
    translateclass: (dictionary) If provided, all class labels are translated using this dict.
    '''
    d=Training(fname)
    
    f=open(os.path.abspath(os.path.expanduser(fname)), 'rU')
    
    try:
        for l in f:
            line=l.strip()
            if line:
                parts = line.split(FS)
                klass = parts[classindex]
                
                if translateclass:
                    try:
                        klass = translateclass[klass]
                    except KeyError:
                        print >>sys.stderr, 'ERROR: training class label %s not present in translatefile option -T' %klass
                        raise
                
                try:
                    d[klass]+=1
                except KeyError:
                    d[klass] = 1
    finally:
        f.close()
    
    return d
    


class ConfusionMatrix(object):
    '''A confusionmatrix. 
    
    Use the update() method to add values.'''
    def __init__(self, training={}):
        '''
        training: the Training dict
        
        A Training dict is a dictionary containing
        a count for every possible class label. 
          E.g. {'label1':39, 'label2':4568}
        '''
        self._classes={}
        if not training: training = {}
        self._training=training

        
    def __len__(self):
        '''The total number of instances'''
        total=0
        for value in self._classes.values():
            total += sum(value.values())
        return total
        
    def __repr__(self):
        return '<ConfusionMatrix object based on %d instances>' %len(self)
        
    def __str__(self):
        if not self.classes:
            raise EmptyError()
        
        try:
            #All classes (also those that were only predicted but never in the reference!)
            keys = set(self._classes.keys())
            for v in self._classes.values():
                keys.update(set(v.keys()))
                
            keys = list(keys)
            keys.sort()
                
            #Add a line of data as should be printed in the confusion matrix
            values=[]
            for key in keys:
                linedata=[]
                d=self._classes.get(key, {})
                for k in keys:
                    linedata.append(d.get(k, 0))
                values.append(tuple([str(key)]+linedata))
                
            #The format of one line
            #the longest classname
            length = max([len(str(n)) for n in keys]) + 2
            
            valueformat = ' | %'+str(length) +'d'
            format = '%'+ str(length) + 's'+ valueformat*len(keys)
            
            output=[]
            
            #headers
            classformat = ' | %'+str(length) +'s'
            firstlineformat = '  '+ ' '*length + classformat*len(keys)
            output.append(firstlineformat %tuple(keys))
            #a border
            output.append('  '+'-'*len(output[0]))
            predformat = '  %'+str(len(output[0]))+'s'
            output.insert(0, predformat %('predicted'))
            
            #the ref label
            for i, line in enumerate(values):
                l = format %line
                if i == 0:
                    l = 'r '+l
                elif i == 1:
                    l = 'e '+l
                elif i == 2:
                    l = 'f '+l
                else:
                    l = '  '+l
                output.append(l)
            
            diff = -1*(len(values) - 3)
            
            if diff > 0:
                for i in range(diff):
                    output.append('ref'[diff - 1 +i])
            
            return '\n'.join(output)
        except MemoryError:
            return "Couldn't print confusionmatrix because of limited memory"
            
    @property
    def classes(self):
        '''The classes in the matrix'''
        classes = self._classes.keys()[:]
        classes.sort()
        return classes
        
    @property
    def training(self):
        '''The class distribution of training'''
        if not self._training: raise TrainingError
        return self._training
        
    @property
    def accuracy(self):
        '''Returns the overal accuracy'''
        if not self.classes:
            raise EmptyError()
        return sum([self.TP(c) for c in self.classes]) / float(len(self))
            
    def update(self, ref, pred):
        '''
        Updates the confusionmatrix.
        ref: the reference class
        pred: the predicted class
        '''
        self._classes[ref][pred] = self._classes.setdefault(ref, {}).setdefault(pred, 0) + 1
        
    def getattr(self, att):
        return self.__getattribute__(att)
        
    def reset(self):
        '''Resets the matrix'''
        self._classes={}
        
    def count(self, label):
        '''Returns the number of occurrences of the class label in the reference.'''
        if not self.classes:
            raise EmptyError()
        try:
            return sum(self._classes[label].values())
        except KeyError:
            raise ClassError(label)
        
    def TP(self, label):
        '''Returns the true positive count for class label.'''
        if not self.classes:
            raise EmptyError()
        try:
            return self._classes[label][label]
        except KeyError:
            return 0
            
    def FP(self, label):
        '''Returns the false positive count for class label.'''
        if not self.classes:
            raise EmptyError()
        if label not in self.classes:
            return 0
        fp=0
        for k,v in self._classes.items():
            if k != label:
                fp+=v.get(label, 0)
        return fp
        
    def FN(self, label):
        '''Returns the false negative count for class label'''
        if not self.classes:
            raise EmptyError()
        if label not in self.classes:
            return 0
        fn = 0
        for k,v in self._classes[label].items():
            if k != label:
                fn+=v
        return fn
        
    def TN(self, label):
        '''Returns the true negative count for class label'''
        if not self.classes:
            raise EmptyError()
        if label not in self.classes:
            raise ClassError(label)
        return len(self) - self.TP(label) - self.FP(label) - self.FN(label)
        
    def precision(self, label):
        '''Return the precision of the class label'''
        tp = float(self.TP(label))
        fp= self.FP(label)
        
        try:
            return tp/(tp+fp)
        except ZeroDivisionError:
            return 0
        
    def recall(self, label):
        '''Returns the recall of the class label'''        
        tp = float(self.TP(label))
        
        if tp == 0:
            return 0
        
        fn= self.FN(label)
        return tp/(tp+fn)
        
    def fmeasure(self, label, beta=1.0):
        '''
        Returns the fmeasure with beta for the class label:
        (1.0 + beta) * (P*R) / (beta*P + R)
        '''
        P = self.precision(label)
        R = self.recall(label)
        try:
            return (1.0 + beta) * (P*R) / (beta*P + R)
        except ZeroDivisionError:
            return 0
        
    def macrofmeasure(self, beta=1.0):
        '''The macroaveraged fmeasure'''
        if not isinstance(beta, float):
            raise TypeError('beta must be a float or integer')
            
        try:
            return sum([self.fmeasure(c, beta=beta) for c in self.classes])/len(self.training)
        except TrainingError:
            return sum([self.fmeasure(c, beta=beta) for c in self.classes])/len(self.classes) 
        
        
    def microfmeasure(self, beta=1.0):
        '''The microaveraged fmeasure.'''
        if not isinstance(beta, float):
            raise TypeError('beta must be a float or integer')
            
        try:
            return sum([self.fmeasure(c, beta=beta)*self.training[c]/sum(self.training.values()) for c in self.training.keys()])
        except TrainingError:
            return sum([self.fmeasure(c, beta=beta)*self.count(c)/len(self) for c in self.classes])
            
            
    def macrorecall(self):
        '''The macroaveraged recall'''
        try:
            return sum([self.recall(c) for c in self.classes])/len(self.training)
        except TrainingError:
            return sum([self.recall(c) for c in self.classes])/len(self.classes) 
            
    def macroprecision(self):
        '''The macroaveraged precision'''
        try:
            return sum([self.precision(c) for c in self.classes])/len(self.training)
        except TrainingError:
            return sum([self.precision(c) for c in self.classes])/len(self.classes) 
        

    def microrecall(self):
        '''The microaveraged recall.'''            
        try:
            return sum([self.recall(c)*self.training[c]/sum(self.training.values()) for c in self.training.keys()])
        except TrainingError:
            return sum([self.recall(c)*self.count(c)/len(self) for c in self.classes])
    def microprecision(self):
        '''The microaveraged precision.'''            
        try:
            return sum([self.precision(c)*self.training[c]/sum(self.training.values()) for c in self.training.keys()])
        except TrainingError:
            return sum([self.precision(c)*self.count(c)/len(self) for c in self.classes])
        
                
#SHELL SCRIPT++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _main(fname, fs, beta, training=None, printcm=True, translateclass=None):
    cm = getConfusionMatrixFromFile(fname, fs, training=training, translateclass=translateclass)
    if printcm:  
        print cm
        print
        
    #Calcutalte the length of the longest classlabel
    length= max([len(str(c)) for c in cm.classes])
    
    print ('%-'+str(length)+'s ') %(' '),'%5s  %5s  %5s  %5s' %('TP', 'FP', 'TN', 'FN')
    for c in cm.classes:
        print ('%-'+str(length)+'s:') %str(c),'%5d  %5d  %5d  %5d' %(cm.TP(c), cm.FP(c), cm.TN(c), cm.FN(c)) 
    print
    lineformat='precision: %6.4f  recall:  %6.4f   F-measure(beta=%3.1f): %6.4f'
    for c in cm.classes:
        print ('%-'+str(length)+'s:') %str(c), lineformat %(cm.precision(c), cm.recall(c), beta, cm.fmeasure(c, beta))
    print
        
        
    output=[]
    if training:
        output.append('Computing averaged scores using train (like Timbl):')
    else:
        output.append('Computing averaged scores using test (may differ slightly from Timbl):')
    #microrecall = sum([cm.recall(cl)*cm.count(cl)/len(cm) for cl in cm.classes])
    #microprec = sum([cm.precision(cl)*cm.count(cl)/len(cm) for cl in cm.classes])
    #output.append( 'Microaveraged recall             : %6.4f' %microrecall)
    #output.append( 'Microaveraged precision          : %6.4f' %microprec)

    output.append( 'Macroaveraged F-measure(beta=%3.1f): %6.4f' %(beta, cm.macrofmeasure(beta)) )
    output.append( 'Microaveraged F-measure(beta=%3.1f): %6.4f' %(beta, cm.microfmeasure(beta)) )
    output.append( 'Overall accuracy                 : %6.4f (%d out of %d)' %(cm.accuracy, sum([cm.TP(c) for c in cm.classes]), len(cm)) )
    
    print '\n'.join(output)
    #return '\n'.join(output)


def _main2(fname, classfile, classindex, fs, beta, printcm, training, cutoff, translateclass=None):
    testtmp = reformat(fname, classfile, classindex=classindex, FS=fs, cutoff=cutoff)

    try:
        _main(testtmp, fs, beta, training=training, printcm=printcm, translateclass=translateclass)
    except:
        os.remove(testtmp)
        raise
    else:
        os.remove(testtmp)

def _usage():
    print '''Prints the metrics of a machine learning testfile (version %s)
USAGE:
    ./confusionmatrix.py [-f FS] [-b beta] [-T translation] [-t trainingfile] [-c classindex] [-C cutoff] file [classfile]
    
ARGUMENTS:
        file:   a list of tagged instances with at the second last place the reference class and
                at the last place the predicted value (default Timbl)
    OR
        file:       a list of instances with at classindex the reference class
        classfile:  a list of predicted classlabels
        The number of lines in file and classfile must be the same!
        (default maxent)
             
OPTIONS
    -f FS:   FS is the field separator, defaults to all whitespace. (--field-separator)
    -t trainingfile: For Timbl files you can specify the trainingfile. If given, the script
                     uses this file to compute averaged scores. See NOTE. It is assumed
                     that the last field is the classlabel.
    -b beta: the beta value used to calculate the F-measures. (default:1)(--beta)
    -c classindex:  the index of the reference classlabel in file. Starting with 0. (default:the last)
                    Only useful if a classfile is specified. (--class)
    -C cutoff: This option can be used for SVMLight files. All values in the given classfile strictly below
               the cutoff get class label -1. The others get class label 1. Normally, this is set to 0 for
               SVM files. (--cutoff)
    -m : Don't print the confusionmatrix. Useful when there are a lot of classes. (--confusionmatrix)
                    
    -T translation: A file with the format:
                        old_label > new_label
                        ...
                    When provided, the class labels as they are in training, class and output file are
                    translated into the new labels.
                    
EXAMPLES
    1. TiMBL
    +++++++++++
    To use on a Timbl-out file:
        ./confusionmatrix.py -f ',' -t dimin.train dimin.test.IB1.O.gr.k1.out
    
    2. MAXENT
    +++++++++++
    To use on a maxent test- and outputfile:
        ./confusionmatrix.py -c 0 testfile outputfile
        
    The outputfile is the file created with the -o option of maxent.
    
    3. SVMLight (binary)
    +++++++++++++++++++++
    To use on SVMLight test- and outputfile:
        ./confusionmatrix.py -c 0 -C 0 tesfile outputfile
        
    Note that this only works for binary SVM. The class labels should be -1 or 1.
        
    
NOTE
    The macro-averaged and micro-averaged f-scores (without the -t option) may differ from
    the scores that are given by Timbl when using the +v cs option. The -t option is there
    to solve this issue.
    
    Timbl uses the number of different classes in training for the macro-averaged scores.
    This script uses the number of different classes in test. Therefore, if a class occurs
    in training and not in test the macro-averaged scores of Timbl and this script will differ.
    A difference between Timbl and this script will not be observed very often since most of
    the time all classes of training are also in test and vice versa. Using the -t option will
    resolve this difference.
    
    Timbl uses the frequency of the classes in training when computing micro-averaged scores whereas
    this script uses the frequencies in test. Because the frequencies in training and test often differ
    the micro-averaged scores of Timbl and this script will also differ often. Using the -t option will
    resolve this difference.
    

%s, %s
''' %(__version__, __author__, __date__)
        
if __name__ == '__main__':
    import getopt
    
    try:
        opts,args=getopt.getopt(sys.argv[1:],'f:hb:c:t:mC:T:', ['help', 'confusionmatrix', 'field-separator=', 'beta=', 'class=', 'train=', 'cutoff=', 'translate='])
    except getopt.GetoptError:
        # print help information and exit:
        _usage()
        sys.exit(2)

    FS=None
    beta=1.0
    classindex=-1
    trainingfile=None
    printcm=True
    cutoff=None
    translatefile = None
    
    for o, a in opts:
        if o in ('-h', '--help'):
            _usage()
            sys.exit()
        if o in ('-f', '--field-separator'):
            FS=a
        if o in ('-t', '--train'):
            trainingfile = os.path.abspath(os.path.expanduser(a))
        if o in ('-T', '--translate'):
            translatefile = os.path.abspath(os.path.expanduser(a))
        if o in ('-m', '--confusionmatrix'):
            printcm=False
        if o in ('-b', '--beta'):
            try:
                beta=float(a)
            except ValueError:
                print >>sys.stderr, 'Error: beta must be a float.'
                sys.exit(1)
        if o in ('-c', '--class'):
            try:
                classindex=int(a)
            except ValueError:
                print >>sys.stderr, 'Error: classlabel index must be an integer.'
                sys.exit(1)
        if o in ('-C', '--cutoff'):
            try:
                cutoff=float(a)
            except ValueError:
                print >>sys.stderr, 'Error: cutoff must be a float.'
                sys.exit(1)

            
    #Get control characters for FS
    if FS:
        FS = FS.replace('\\t', '\t')
        FS = FS.replace('\\n', '\n')
        FS = FS.replace('\\r', '\r')
            
        
    # 1 or 2 arguments needed
    classfile=None
    if len(args) == 2:
        classfile=os.path.abspath(os.path.expanduser(args[1]))
    elif len(args) > 2:
        _usage()
        sys.exit(2)
    elif not args:
        _usage()
        sys.exit(2)
    
    fname = os.path.abspath(os.path.expanduser(args[0]))
    
    if not os.path.isfile(fname):
        print >>sys.stderr, 'Error: %s is not an existing file' %fname
        sys.exit(1)
    
    if trainingfile and not os.path.isfile(trainingfile):
        print >>sys.stderr, 'Error: %s is not an existing file' %trainingfile
        sys.exit(1)
        
    if classfile and not os.path.isfile(classfile):
        print >>sys.stderr, 'Error: %s is not an existing file' %classfile
        sys.exit(1)
        
    if translatefile and not os.path.isfile(translatefile):
        print >>sys.stderr, 'Error: %s is not an existing file' %translatefile
        sys.exit(1)
        
    # translation
    translateclass=None
    if translatefile:
        translateclass = dict([(k.strip(), v.strip()) for k,v in _read(translatefile, '>')])
        translateclass = Translate(translatefile, translateclass)
    
    # training 
    training=None
    if trainingfile:
        training = getTraining(trainingfile, classindex=classindex, FS=FS, translateclass=translateclass)
        
    try:
        if classfile:
            _main2(fname, classfile, classindex, FS, beta, printcm, training, cutoff, translateclass=translateclass)
        else:
            _main(fname, FS, beta, training=training, printcm=printcm, translateclass=translateclass)
    except Error, e:
        print >>sys.stderr, 'Error:', e.message
        sys.exit(1)

    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
