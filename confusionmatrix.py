#!/usr/bin/env python
# -*- coding: utf8 -*-
'''A script to compute evaluation statistics from files

Example usage inside a script:
    
The Timbl commando "Timbl -f dimin.train -t dimin.test +v cs+cm" gave an outputfile called : dimin.test.IB1.O.gr.k1.out
    >>> cm = getConfusionMatrixFromFile('dimin.test.IB1.O.gr.k1.out', fsep=',')
    >>> print cm
                                                           predicted
                         E         J         K         P         T
                -------------------------------------------------------
    g        E |        87         4         8         1         0
    o        J |         5       346         0         0         1
    l        K |         7         0         9         0         0
    d        P |         3         0         0        24         0
             T |         0         2         0         0       453

    >>> print cm.averaged(level=MACRO, score=fscore)
    0.861914364513

For more info:
    >>> help(cm)


Based on:
- Grigorios Tsoumakas, Ioannis Katakis, and Ioannis Vlahavas, Mining Multi-label Data, Data Mining and Knowledge Discovery Handbook, Part 6, O. Maimon, L. Rokach (Ed.), Springer, 2nd edition, pp. 667-685, 2010.
- Daelemans, W., Zavrel, J., van der Sloot, K., & van den Bosch, A. (2010). Timbl: Tilburg memory-based learner, version 6.3. Tech. Rep. ILK 10-01, Tilburg University
- van Rijsbergen, C. J. (1975). Information Retrieval . London, UK: Butterworths.

For more informations about the computation of the averaged scores:
http://www.clips.ua.ac.be/~vincent/pdf/microaverage.pdf


# License: GNU General Public License, see http://www.clips.ua.ac.be/~vincent/scripts/LICENSE.txt
'''
__date__ = 'April 2013'
__version__ = '2.1.4'
__author__ = 'Vincent Van Asch'


import os, sys, getopt, re
from math import pow

# CONSTANTS ##############################################################################

MICRO=0
MACRO=1
MICROt=2
UTT = re.compile('<utt>')
encoding = 'utf-8'

FSCORE='fscore'
PRECISION='precision'
RECALL='recall'

# CODE ###################################################################################

def kwad(x): return pow(x, 2)

def precision(tp, fp, fn, beta=None):
    if tp == 0: return 0
    return float(tp)/(tp + fp)
def recall(tp, fp, fn, beta=None):
    if tp == 0: return 0
    return float(tp)/(tp + fn)
def fscore(tp, fp, fn, beta=1):
    if not isinstance(beta, (int, float)): raise TypeError('beta must be a float or integer')
    p = precision(tp, fp, fn)
    r = recall(tp, fp, fn)
    beta = float(beta)
    if p == 0 or r == 0: return 0
    return ((1.0+kwad(beta))*p*r)/(kwad(beta)*p+r) 

class ConfusionMatrix(dict):
    '''A confusion matrix that can handle multilabels'''
    def __init__(self, freq={}, encoding=encoding, compute_none=False, strict=False, cached=True):
        '''freq        : the counts of the labels in the training corpus as a dictionary
                             key: label
                             value: count
           encoding    : encoding to print the labels. Note that the input labels should
                         be unicode.
           compute_none: if True, the None label is taken into account for the label frequences.
                         This has an influence on the macro-averaged scores and the
                         label-freq-based micro-averaged scores.
           strict      : if True, only use labels from the gold-standard (and not from predicted)
           cached      : if set to True, scores are cached. This means that after computing a score, using the add()
                         or add_training() method may introduce errors.
        '''
        dict.__init__(self, {})
        
        # To store information about the training corpus
        self._freq=freq
        
        # For the accuracy
        self.total=0
        self.correct=0
        
        self.encoding = encoding
        self.compute_none=compute_none
        self.strict=strict
        
        self.cached={}
        if cached:  self.cached={1:1}
        
    def __str__(self):
        N = len(self.observed_labels)
        
        # A None safe version of the labels
        labels=[]
        for l in self.observed_labels:
            if l is None:
                labels.append('*')
            else:
                labels.append(l)

        # The width of the columns
        s = max([max([0]+[len(l) for l in labels if isinstance(l, (str, unicode))]), 6])
        
        format = '    '.join(['%%%ds' %(s+2)]+['%%%ds' %(s) for i in range(N)])
        format = '%s   '+format
        
        # The first lines
        lines=[format %tuple(['' for i in range(N+1)]+['predicted'])]
        lines.append(format %tuple([' ', ' ']+labels))
        lines.append('            '+'-'*(len(lines[-1])-7))
        
        if len(labels) > 13:
            caption=list('gold-standard')
        else:
            caption = list('gold')
        
        # Adding the data
        for glabel in self.observed_labels:
            if glabel is None:
                line=['* |']
            else:
                line = [unicode(glabel)+' |']
            for plabel in self.observed_labels:
                if glabel in self.keys():
                    v = self[glabel].get(plabel, 0)
                else:
                    v=0
                
                if glabel is None and plabel is None:
                    # predicting empty position as empty position makes no sense, so don't put a number
                    line.append('-')
                else:
                    if not v%1:
                        line.append('%.0f' %v)
                    else:
                        line.append('%.2f' %v)
                      
            try:
                letter = caption.pop(0)
            except IndexError:
                letter=' '
                            
            lines.append(format %tuple([letter]+line))
            
        return ('\n'.join(lines)).encode(self.encoding)
    
    def single_add(self, g, p):
        '''Takes two labels: a gold and a predicted label and
        adds them to the confusionmatrix.
        
        Faster than the add() method, but cannot handle multi-labels.
        '''
        # For accuracy
        if g == p: self.correct+=1
        self.total+=1
    
        # Fill
        self[g][p] = self.setdefault(g, {}).setdefault(p, 0) + 1
    
    
    def add(self, g, p):
        '''Takes a list with gold labels and one with predicted labels
        and adds them to the matrix.
        
        Each instance in the test corpus should be added with this method.'''
        if not isinstance (g, list): g = [g]
        if not isinstance (p, list): p = [p] 
        
        # For safety, because we remove elements
        gold = g[:]
        pred = p[:]
        
        # For the accuracy
        if set(gold) == set(pred): self.correct+=1
        self.total+=1
        
        # The correctly predicted ones
        cache=[]
        for g in gold:
            if g is not None:
                if isinstance(g, str):
                    g = g.decode(self.encoding)
                elif isinstance(g, (unicode, int, float)):
                    pass
                else:
                  raise TypeError('labels should be unicode objects')
            if g in pred:
                if g not in self.keys():
                    self[g]={g:1}
                else:
                    self[g][g] = self[g].get(g, 0) + 1
                
                pred.remove(g)
            else:
                cache.append(g)
        gold=cache[:]

        # the wrong and missing predictions
        if len(gold) <= len(pred):
            # If we have less golds than predictions 
            while len(gold) < len(pred): gold.append(None)
        else:
            # if we have less predictions than golds
            while len(pred) < len(gold): pred.append(None)
            
        if pred:
            # Calculate the value
            point = 1.0/len(pred)
            for g in gold:
                for p in pred:
                    if p is not None:
                        if isinstance(p, str):
                            p = p.decode(self.encoding)
                        elif isinstance(p, (unicode, int, float)):
                            pass
                        else:
                          raise TypeError('labels should be unicode objects')
                          
                    if g not in self.keys():
                        self[g]={p:point}
                    else:
                        self[g][p] = self[g].get(p, 0) + point

        
    def add_training(self, g):
        '''Adds a list of labels from the training corpus.
        This is used when computing averaged scores with level MICROt.
        
        Each instance from the training corpus should be added with this method.
        '''
        for l in g:
            if isinstance(l, str):
                l = l.decode(self.encoding)
            elif isinstance(l, (unicode, int, float)):
                pass
            else:
              raise TypeError('labels should be unicode objects')

            self._freq[l] = self._freq.get(l, 0) + 1

    @property
    def labels(self):
        '''Returns all the labels'''
        if 'labels' not in self.cached.keys():
            out=set(self.keys())
            
            if not self.strict:
                for v in self.values():
                    out.update(set(v.keys()))
            out = list(out)
            out.sort()
            
            # Put None in last position
            if None in out:
                out.remove(None)
                out.append(None)
            
            if self.cached:
                self.cached['labels']=out
            
        if 'labels' in self.cached.keys(): return self.cached['labels']
        return out
            
    @property
    def observed_labels(self):
        '''Returns all labels that are observed in the test corpus, 
        irrespective of the fact if it comes from gold-standard or prediction'''
        if 'observed_labels' not in self.cached.keys():
            out=set(self.keys())
            
            for v in self.values():
                out.update(set(v.keys()))
            out = list(out)
            out.sort()
            
            # Put None in last position
            if None in out:
                out.remove(None)
                out.append(None)
            
            if self.cached:
                self.cached['observed_labels']=out
            
        if 'observed_labels' in self.cached.keys(): return self.cached['observed_labels']
        return out
            
    @property
    def freq_training(self):
        '''The relative frequencies of label in the training corpus.
        
        Note that the None label is not included because it is not a true label.
        '''
        if 'freq_training' not in self.cached.keys():
            # The total number of labels
            if self.compute_none:
                total = sum(self._freq.values())
            else:
                total=sum([v for k,v in self._freq.items() if k is not None])
            
            # The relative frequencies
            out={}
            for l, v in self._freq.items():
                if not self.compute_none:
                    if l is None: continue
                out[l] = float(v)/total
                
            if self.cached: self.cached['freq_training']=out
                
        if 'freq_training' in self.cached.keys(): return self.cached['freq_training']
        return out
            
    @property
    def freq_test(self):
        '''The relative frequencies of labels in the test corpus.
        Note that the None label is not included because it is not a true label.
        '''
        if 'freq_test' not in self.cached.keys():
            out={}
                
            total = 0
            for l in self.labels:
                if not self.compute_none:
                    if l is None: continue
                out[l] = float(self.tp(l)+self.fn(l))
                total += (self.tp(l)+self.fn(l))
            for k, v in out.items():
                out[k] = v/total
                
            if self.cached: self.cached['freq_test'] = out

        if 'freq_test' in self.cached.keys(): return self.cached['freq_test']
        return out
            
    @property
    def ng(self):
        '''Returns the number of different labels in the gold-standard of test'''
        o = self.keys()
        if not self.compute_none:
            if None in o: o.remove(None)
        return len(o)
        
        
    @property
    def np(self):
        '''Returns the number of different labels in the predictions of test'''
        out=set()
        for v in self.values():
            out.update(set(v.keys()))
        out=list(out)
            
        if not self.compute_none:
            if None in out: out.remove(None)
        return len(out)
        
    @property
    def nG(self):
        '''Returns the number of different labels in the training corpus'''
        return len(self.freq_training.keys())
        
            
    def tp(self, label):
        '''Returns the true positive of a given label'''
        if label in self.keys():
            return self[label].get(label, 0)
        else:
            return 0
            
    def fp(self, label):
        '''Returns the false positives of a label'''
        cache_key = u'fp_'+unicode(label)
        if cache_key not in self.cached.keys():
            fp=0
            for l in self.labels:
                if l != label:
                    if l in self.keys():
                        fp += self[l].get(label, 0)
                        
            if self.cached: self.cached[cache_key]=fp
        
        if cache_key in self.cached.keys(): return self.cached[cache_key]
        return fp
                 
    def fn(self, label):
        '''Returns the false negatives of a label'''
        cache_key = u'fn_'+unicode(label)
        if cache_key not in self.cached.keys():
            fn=0
            if label in self.keys():
                for l, v in self[label].items():
                    if l != label:
                        fn+=v
                        
            if self.cached: self.cached[cache_key]=fn
        
        if cache_key in self.cached.keys(): return self.cached[cache_key]
        return fn
        
    def precision(self, label):
        '''Returns the precision of a label'''
        return precision(self.tp(label), self.fp(label), 0)
        
    def recall(self, label):
        '''Returns the recall of a label'''
        return recall(self.tp(label), 0, self.fn(label))
        
    def fscore(self, label, beta=1):
        '''Returns the F-score for a label'''
        return fscore(self.tp(label), self.fp(label), self.fn(label), beta=beta)
        
    def accuracy(self):
        '''The accuracy
        In the case of multilabels: correct means the whole label the same'''
        return float(self.correct)/self.total
        
    def averaged(self, level=MICRO, score=FSCORE, training=False, beta=1):
        '''Returns the averaged version of the score
        
        Possible levels:
            MICRO: micro-averaging: score( sum_q[tp_i], sum_q[fp_i], sum_q[fn_i]) 
            MACRO: macro-averaging: 1/q * sum_q[ score(tp_i, fp_i, fn_i) ]
            MICROt: micro-averaging: sum_q[ f_i*score(tp_i, fp_i, fn_i) ]
        
            with:
                score()          : a given evaluation score
                tp_i, fp_i, fn_i : count for label i
                f_i              : relative frequency of label in in training corpus
                q                : the number of different labels
                sum_q            : summing over all labels
                
        score: one of the following values: PRECISION, RECALL, FSCORE
        training: use relative frequencies and labels from the training corpus (obligatory for MICROt)
        
        beta: the beta value if fscore is computed
        '''
        # The labels in training
        if training:
            labels = self.freq_training.keys()
        else:
            labels = self.labels[:]

        if not self.compute_none:
            if None in labels: labels.remove(None)
        elif None not in labels:
            labels.append(None)

        if score == FSCORE:
            func = fscore
        elif score == PRECISION:
            func = precision
        elif score == RECALL:
            func = recall
        else:
            raise ValueError('Unknown score: %s' %score)

        if level == MACRO:            
            return sum([func( self.tp(l), self.fp(l), self.fn(l), beta ) for l in labels]) / float(len(labels)) 
        elif level == MICRO:
            return func(sum([self.tp(l) for l in labels]), sum([self.fp(l) for l in labels]), sum([self.fn(l) for l in labels]), beta=beta)
        elif level == MICROt:
            if training:
                # Use the frequencies from the training corpus
                return sum([self.freq_training.get(l, 0)*func( self.tp(l), self.fp(l), self.fn(l), beta=beta ) for l in labels])
            else:
                # Use the frequencies from the test corpus
                return sum([self.freq_test.get(l, 0)*func( self.tp(l), self.fp(l), self.fn(l), beta=beta ) for l in labels])
            
        else:
            raise ValueError('Unknown level: %s' %str(level))
        

distribution = re.compile('{[^}]+}')
def fread(fname, encoding=encoding, delete_distr=True):
    '''delete_distr: if True remove {...} at the end of the line'''
    with open(os.path.abspath(os.path.expanduser(fname)), 'rU') as f:
        for l in f:
            if delete_distr: l = distribution.sub('', l)
            line = l.strip()
            if line: yield line.decode(encoding)    

def get_translation(fname, encoding=encoding):
    out={}
    for line in fread(fname, encoding=encoding):
        k, v = line.split('>')
        out[k.strip()] = v.strip() 
    return out

def get_plain_iterator(fname, gold_index, pred_index=None, fsep=None, lsep='_', encoding=encoding, multi=False, ignore=[]):
    '''Yields all gold_labels, pred_lables tuples
     If pred_index is None: it only looks for gold labels (training files, maxent files, ...)
     
    gold_index: the index of the gold standard label
    pred_index: the index of the predicted label
     
    fsep: the field separator
    lsep: in the case of multilabels, the label separator
    
    multi: if True, split labels using lsep
    
    encoding: encoding used to decode the files
    '''
    if isinstance(fsep, str): fsep=fsep.decode(encoding)
    if isinstance(lsep, str): lsep=lsep.decode(encoding)
    
    for line in fread(fname, encoding=encoding):
        # Don't keep lines that match
        if ignore:
            for pattern in ignore:
                quit=False
                if pattern.search(line):
                    quit=True
                    break
            if quit: continue

        # The fields
        fields = line.split(fsep)
        
        # GOLD
        try:
            gold = fields[gold_index]
        except IndexError:
            raise IndexError('Maybe the field separator is not set correctly, see -f') 
        if multi:
            gold = gold.split(lsep)
        else:
            gold = [gold]
            
        if pred_index is not None:
            pred = fields[pred_index]
            if multi:
                pred = pred.split(lsep)
            else:
                pred = [pred]
                
        if pred_index is None:
            yield gold
        else:
            yield gold, pred
            
        
def get_combination_iterator(test_file, prediction_file, gold_index, fsep=None, lsep='_', encoding=encoding, multi=False, cutoff=None, ignore=[], pos_label='+1', neg_label='-1'):
    test = fread(test_file, encoding=encoding)
    pred = fread(prediction_file, encoding=encoding)

    for line, pred in zip(test, pred):
        # Don't keep lines that match
        if ignore:
            for pattern in ignore:
                quit=False
                if pattern.search(line):
                    quit=True
                    break
            if quit: continue
    
    
        gold = line.split(fsep)[gold_index]
        
        if cutoff is None:
            # Maxent style
            if multi:
                gold = gold.split(lsep)
                pred = pred.split(lsep)
            else:
                gold = [gold]
                pred = [pred]
        else:
            # SVM style
            if multi: raise ValueError('Do not know how to handle multi labels together with cutoff')
            
            gold=[gold]
            
            if float(pred) >= cutoff:
                pred = [pos_label]
            else:
                pred = [neg_label]
            
        yield gold, pred
        
def getConfusionMatrixFromIterator(iterator, training_iterator=None , encoding=encoding, translation=None, compute_none=False, strict=False):
    '''Construct a confusion matrix from an iterator
    
    ignore: a list of Pattern objects. Lines that match any of these are omitted. 
    multi: set to True if the labels are multilabels and should be analyzed at the single label level
    training_iterator: produce labels in train
    
    translation: a dict. All labels are translated key => value.
    compute_none: if True, the None label is taken into account for the label frequences. This has an influence on
                  the macro-averaged scores and the label-freq-based micro-averaged scores.
    '''
    cm = ConfusionMatrix(encoding=encoding, compute_none=compute_none, strict=strict)
                
    if training_iterator is not None:
        for labels in training_iterator:
            if translation:
                try:
                    labels = [translation[l] for l in labels]
                except KeyError, e:
                    raise KeyError('"%s" is not present in translation' %(e.args[0].encode(encoding)))
        
            cm.add_training(labels)
        
    for gold_labels, pred_labels in iterator:
        if translation:
            try:
                gold_labels = [translation[l] for l in gold_labels]
                pred_labels = [translation[l] for l in pred_labels]
            except KeyError, e:
                raise KeyError('"%s" is not present in translation' %(e.args[0].encode(encoding)))
        cm.add(gold_labels, pred_labels)
        
    return cm
    
def getConfusionMatrixFromFile(fname, gold_index=-2, pred_index=-1, fsep=None, lsep='_', ignore=[], multi=False, training_file=None, train_gold_index=-1, encoding=encoding, strict=False, compute_none=False, cached=True):
    '''Construct a confusion matrix from a TiMBL-style file
    
    gold_index      : the index of the gold standard label in test
    pred_index      : the index of the predicted label in test
    train_gold_index: the index of the gold standard label in the optional training file
     
    fsep: the field separator
    lsep: in the case of multilabels, the label separator
    
    ignore: a list of Pattern objects. Lines that match any of these are omitted. 
    multi: set to True if the labels are multilabels and should be analyzed at the single label level
    training_file: path to the training file used to produce the predictions
    
    compute_none: if True, the None label is taken into account for the label frequences.
                  This has an influence on the macro-averaged scores and the
                  label-freq-based micro-averaged scores.
    strict      : if True, only use labels from the gold-standard (and not from predicted)
    cached      : if set to True, scores are cached. This means that after computing a score, using the add()
                  or add_training() method may introduce errors.
    '''
    cm = ConfusionMatrix(strict=strict, compute_none=compute_none, cached=cached, encoding=encoding)
        
    training_iterator=None
    if training_file:
        training_iterator = get_plain_iterator(training_file, train_gold_index, pred_index=None, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi)

    iterator = get_plain_iterator(fname, gold_index, pred_index=pred_index, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, ignore=ignore) 
   
    return getConfusionMatrixFromIterator(iterator, training_iterator=training_iterator, encoding=encoding)
        
                
def main(test_file, gold_index=-2, pred_index=-1, fsep=None, lsep='_', ignore=[], multi=False, training_file=None, train_gold_index=-1, beta=1, print_cm=True,\
         verbose=True, prediction_file=None, cutoff=None, translation_file=None, compute_none=False, strict=False, print_labels=True,
         pos_label='+1', neg_label='-1'):
         
    # Getting the optional translation
    translation=None
    if translation_file: translation = get_translation(translation_file)
    
    # Creating the suitable iterators
    training_iterator=None
    if prediction_file:
        # SVM or Maxent: test file and prediction file are separate
        iterator = get_combination_iterator(test_file, prediction_file, gold_index, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, cutoff=cutoff, ignore=ignore, pos_label=pos_label, neg_label=neg_label)
    else:
        # TiMBL: test file and prediction file are one
        iterator = get_plain_iterator(test_file, gold_index, pred_index=pred_index, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, ignore=ignore)
        
    # Create the CM
    if training_file: training_iterator = get_plain_iterator(training_file, train_gold_index, pred_index=None, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi)
    cm = getConfusionMatrixFromIterator(iterator, training_iterator=training_iterator, encoding=encoding, translation=translation, compute_none=compute_none, strict=strict)
    
    
    # Size of largest label
    safe_labels = set(cm.freq_test.keys())
    if training_file: safe_labels.update(set(cm.freq_training.keys()))
    safe_labels = list(safe_labels)
    if None in safe_labels: safe_labels.remove(None)
    s = max(6, max([len(l) for l in safe_labels]))+1
    
    # Get the labels to report on
    if training_file:
        labels = cm.freq_training.keys()
    else:
        labels = cm.labels
    
    # Extra info
    if verbose:
        print '\n%s\n' %('='*70)
        print 'STATISTICS'
        
        k=''
        if cm.ng < 10:
            k = cm.keys()
            k.sort()
            if None in k: 
                if compute_none: k.append('*')
                k.remove(None)
            k= ': '+(', '.join(k)).encode(encoding)
        print cm.ng, 'different labels in gold-standard of test corpus', k
                
        k=''
        if cm.np < 10:
            k=set()
            for v in cm.values(): k.update(v.keys())
            k = list(k)
            k.sort()
            if None in k: 
                if compute_none: k.append('*')
                k.remove(None)
            k= ': '+(', '.join(k)).encode(encoding)
        print cm.np, 'different labels in prediction of test corpus   ', k
            
        if training_file:
            k=''
            if cm.nG < 10:
                k=cm.freq_training.keys()
                k.sort()
                k= ': '+(', '.join(k)).encode(encoding)
            print cm.nG, 'different labels in training corpus             ', k
        
        print
        ss='TEST CORPUS'
        if training_file:
            ss='TRAINING CORPUS'
        print 'LABEL FREQUENCIES IN', ss
        for l in labels:
            # The label to print
            pl=l
            
            # Make a difference if None should be included or not
            if not compute_none:
                if l is None: continue
            else:
                if l is None: pl='*'
            ff = '%%%ds : %%.5f' %s
            # Print the frequencies
            if training_file:
                print (ff %(pl, cm.freq_training[l])).encode(encoding)
            else:
                print (ff %(pl.encode(encoding), cm.freq_test[l])).encode(encoding)
        
        print '\n%s\n' %('='*70)
    
    # Print confusionmatrix
    if print_cm:
        print 'CONFUSION MATRIX'
        print cm
        
    if print_labels:
        # The counts for all seen labels
        print '\n%s\n' %('='*70)
        format = '    '.join(['%%%ds' %s for i in range(4)])
        print format %('', 'TP', 'FP', 'FN')
        for l in cm.observed_labels:
            pl=l
            if l is None: pl='*'
            print (format %(pl+':', cm.tp(l), cm.fp(l), cm.fn(l))).encode(encoding)
            
        # precision, recall, fscore per label
        print '\n%s\n' %('='*70)
        format = '    '.join(['%%%ds' %s for i in range(4)])
        print format %('', 'PREC', 'RECALL', 'F-SCORE(%.1f)' %beta)
        for l in labels:
            # We don't have to print the None because it is always zero
            if l is None: continue
            print (format %(l+':', '%.5f' %cm.precision(l), '%.5f' %cm.recall(l), '%.5f' %cm.fscore(l, beta=beta))).encode(encoding)
    
    # averaged scores
    print '\n%s\n' %('='*70)
    print 'MACRO-AVERAGED'
    print 'PRECISION    : %.5f' %(cm.averaged(level=MACRO, score=PRECISION, training=training_file))
    print 'RECALL       : %.5f' %(cm.averaged(level=MACRO, score=RECALL, training=training_file))
    print 'F-SCORE (%2.1f): %.5f' %(beta, cm.averaged(level=MACRO, score=FSCORE, training=training_file, beta=beta))
    
    print
    print 'MICRO-AVERAGED'
    print 'PRECISION    : %.5f' %(cm.averaged(level=MICRO, score=PRECISION, training=training_file, beta=beta))
    print 'RECALL       : %.5f' %(cm.averaged(level=MICRO, score=RECALL, training=training_file, beta=beta))
    print 'F-SCORE (%2.1f): %.5f' %(beta, cm.averaged(level=MICRO, score=FSCORE, training=training_file, beta=beta))
    
    print
    print 'MICRO-AVERAGED USING LABEL FREQUENCIES'
    print 'PRECISION    : %.5f' %(cm.averaged(level=MICROt, score=PRECISION, training=training_file))
    print 'RECALL       : %.5f' %(cm.averaged(level=MICROt, score=RECALL, training=training_file))
    print 'F-SCORE (%2.1f): %.5f' %(beta, cm.averaged(level=MICROt, score=FSCORE, training=training_file, beta=beta))
        
    if verbose:
        print
        if training_file:
            print 'INFO: Computing averaged scores using training corpus'
        else:
            print 'INFO: Computing averaged scores only using test corpus'
        
    print '\n%s\n' %('='*70)
    print 'ACCURACY     : %.5f (%d out of %d)' %(cm.accuracy(), cm.correct, cm.total)
        
    print '\n%s\n' %('='*70)
        
      
def meddling_main(test_file, gold_index=-2, pred_index=-1, fsep=None, lsep='_', ignore=[], multi=False, training_file=None, train_gold_index=-1, beta=1, print_cm=True,\
                  verbose=True, prediction_file=None, cutoff=None, translation_file=None, print_labels=True, force=False,
                  pos_label='+1', neg_label='-1'):
    '''A version of main in which the compute_non and strict options are set'''
    # Getting the optional translation
    translation=None
    if translation_file: translation = get_translation(translation_file)
    
    # For the counts
    # Creating the suitable iterators for MICROt, MACRO
    training_iterator=None
    if prediction_file:
        # SVM or Maxent: test file and prediction file are separate
        iterator = get_combination_iterator(test_file, prediction_file, gold_index, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, cutoff=cutoff, ignore=ignore, pos_label=pos_label, neg_label=neg_label)
    else:
        # TiMBL: test file and prediction file are one
        iterator = get_plain_iterator(test_file, gold_index, pred_index=pred_index, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, ignore=ignore)
        
    # Create the CM
    if training_file: training_iterator = get_plain_iterator(training_file, train_gold_index, pred_index=None, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi)
    cm = getConfusionMatrixFromIterator(iterator, training_iterator=training_iterator, encoding=encoding, translation=translation, compute_none=False, strict=True)
      
    # The micro-averaged should not use strict so reread the files
    # Creating the suitable iterators for MICRO
    training_iterator=None
    if prediction_file:
        # SVM or Maxent: test file and prediction file are separate
        iterator = get_combination_iterator(test_file, prediction_file, gold_index, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, cutoff=cutoff, ignore=ignore, pos_label=pos_label, neg_label=neg_label)
    else:
        # TiMBL: test file and prediction file are one
        iterator = get_plain_iterator(test_file, gold_index, pred_index=pred_index, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi, ignore=ignore)
        
    # Create the CM
    if training_file: training_iterator = get_plain_iterator(training_file, train_gold_index, pred_index=None, fsep=fsep, lsep=lsep, encoding=encoding, multi=multi)
    cm2 = getConfusionMatrixFromIterator(iterator, training_iterator=training_iterator, encoding=encoding, translation=translation, compute_none=True, strict=False)
      
      
    # Size of largest label
    safe_labels = set(cm.observed_labels)
    if training_file: safe_labels.update(set(cm.freq_training.keys()))
    safe_labels = list(safe_labels)
    if None in safe_labels: safe_labels.remove(None)
    s = max(6, max([len(l) for l in safe_labels]))+1
      
    # Get the labels to report on
    if training_file:
        labels = cm.freq_training.keys()
    else:
        labels = cm.labels
    labels.sort()
      
    # Extra info
    if verbose:
        print '\n%s\n' %('='*70)
        print 'STATISTICS'
        
        k=''
        if cm.ng < 10:
            k = cm.keys()
            k.sort()
            if None in k: 
                if compute_none: k.append('*')
                k.remove(None)
            k= ': '+(', '.join(k)).encode(encoding)
        print cm.ng, 'different labels in gold-standard of test corpus', k
                
        k=''
        if cm.np < 10:
            k=set()
            for v in cm.values(): k.update(v.keys())
            k = list(k)
            k.sort()
            if None in k: 
                if compute_none: k.append('*')
                k.remove(None)
            k= ': '+(', '.join(k)).encode(encoding)
        print cm.np, 'different labels in prediction of test corpus   ', k
            
        if training_file:
            k=''
            if cm.nG < 10:
                k=cm.freq_training.keys()
                k.sort()
                k= ': '+(', '.join(k)).encode(encoding)
            print cm.nG, 'different labels in training corpus             ', k
        
        print
        ss='TEST CORPUS'
        if training_file:
            ss='TRAINING CORPUS'
        print 'LABEL FREQUENCIES IN', ss
        
        
        for l in labels:
            # The label to print
            pl=l
            
            # Make a difference if None should be included or not
            #if not multi:
            if l is None: continue
            #else:
            #    if l is None: pl='*'
            
            ff = '%%%ds : %%.5f' %s
            
            # Print the frequencies
            if training_file:
                v = cm.freq_training.get(l, 0)
            else:
                v = cm.freq_test.get(l, 0)
                
            try:
                print (ff %(pl, v)).encode(encoding)
            except:
                raise
        
        print '\n%s\n' %('='*70)

    # Print confusionmatrix
    if print_cm:
        print 'CONFUSION MATRIX'
        print cm2
        
    if print_labels:
        # The counts for all seen labels
        print '\n%s\n' %('='*70)
        format = '    '.join(['%%%ds' %s for i in range(4)])
        print format %('', 'TP', 'FP', 'FN')
        for l in cm.observed_labels:
            pl=l
            if l is None: pl='*'
            print (format %(pl+':', cm.tp(l), cm.fp(l), cm.fn(l))).encode(encoding)
            
        # precision, recall, fscore per label
        print '\n%s\n' %('='*70)
        format = '    '.join(['%%%ds' %s for i in range(4)])
        print format %('', 'PREC', 'RECALL', 'F-SCORE(%.1f)' %beta)
        for l in cm.labels:
            # We don't have to print the None because it is always zero
            if l is None: continue
            print (format %(l+':', '%.5f' %cm.precision(l), '%.5f' %cm.recall(l), '%.5f' %cm.fscore(l, beta=beta))).encode(encoding)
      
      
    # averaged scores
    print '\n%s\n' %('='*70)
    print 'MACRO-AVERAGED'
    print 'PRECISION    : %.5f' %(cm.averaged(level=MACRO, score=PRECISION, training=training_file))
    print 'RECALL       : %.5f' %(cm.averaged(level=MACRO, score=RECALL, training=training_file))
    print 'F-SCORE (%2.1f): %.5f' %(beta, cm.averaged(level=MACRO, score=FSCORE, training=training_file, beta=beta))
        
    print
    print 'MICRO-AVERAGED USING LABEL FREQUENCIES'
    print 'PRECISION    : %.5f' %(cm.averaged(level=MICROt, score=PRECISION, training=training_file))
    print 'RECALL       : %.5f' %(cm.averaged(level=MICROt, score=RECALL, training=training_file))
    print 'F-SCORE (%2.1f): %.5f' %(beta, cm.averaged(level=MICROt, score=FSCORE, training=training_file, beta=beta))  
    

    print
    if force:
        print 'MICRO-AVERAGED'
        print 'PRECISION    : %.5f' %(cm2.averaged(level=MICRO, score=PRECISION, training=training_file, beta=beta))
        print 'RECALL       : %.5f' %(cm2.averaged(level=MICRO, score=RECALL, training=training_file, beta=beta))
        print 'F-SCORE (%2.1f): %.5f' %(beta, cm2.averaged(level=MICRO, score=FSCORE, training=training_file, beta=beta))
    else:
        if round(cm2.averaged(level=MICRO, score=RECALL, training=training_file),9) != round(cm2.averaged(level=MICRO, score=PRECISION, training=training_file),9):
            if training_file:
                raise ValueError('''Normally this error only shows up if there are labels in
the test set that are not in the training set. For single-label this
is impossible. For multi-label this would be possible on the multi-label
level, so you should set option -m to break down the multi-label into
single-labels. If you nevertheless want to evaluate on the multi-level,
you should use option -F.''')
            else:
                raise ImplementationError('Unexpected situation')
        
        # This check is no longer a check is beta = 0.5*(-1 + sqrt(5)) or 0.5*(-1 - sqrt(5)), but this beta value is rather unlikely to occur and we trust that it will not be an error anyway 
        if round(cm2.averaged(level=MICRO, score=FSCORE, training=training_file, beta=beta),9) != round(cm2.averaged(level=MICRO, score=FSCORE, training=training_file, beta=1/(float(beta)+1)),9):
            raise ValueError('%f <> %f' %(cm2.averaged(level=MICRO, score=FSCORE, training=training_file, beta=beta), cm2.averaged(level=MICRO, score=FSCORE, training=training_file, beta=1/(float(beta)+1))))
        print 'MICRO-AVERAGED F-SCORE: %.5f' %(cm2.averaged(level=MICRO, score=FSCORE, training=training_file, beta=beta))
    
    
    if verbose:
        print
        if training_file:
            print 'INFO: Computing averaged scores using training corpus'
        else:
            print 'INFO: Computing averaged scores only using test corpus'
    
    # Accuracy  
    print '\n%s\n' %('='*70)
    print 'ACCURACY     : %.5f (%d out of %d)' %(cm.accuracy(), cm.correct, cm.total)
        
    print '\n%s\n' %('='*70)
      
      
def print_info():
    print >>sys.stderr, '''INFORMATION ON THE CALCULATION
    
1. SINGLE LABEL
    - true positive  (TP) : +1 for a given label, if that label is the same
                            in gold-standard and prediction
    - false positive (FP) : +1 for a given label, if that label occurs in the
                            prediction and not in gold-standard
    - false negative (FN) : +1 for a given label, if that label occurs in the
                            gold-standard and not in the prediction

    Note that an error, always leads to a simultaneous increase of the general FP
    and general FN counts.

2. MULTI LABEL
    - true positive  (TP) : +1 for a given label, if that label occurs in the
                            gold-standard multi-label and the predicted multi-label
    - false positive (FP) : +1 for a given label, if that label occurs in the
                            predicted multi-label and not in gold-standard multi-label
    - false negative (FN) : +1 for a given label, if that label occurs in the
                            gold-standard multi-label and not in the predicted multi-label
    
    Note that a diffent number of labels in the gold-standard and predicted multi-labels
    leads to counting FP and FN for the empty label, indicated with '*'. This ensures
    that an error, always leads to a simultaneous increase of the general FP
    and general FN counts.
    
    In the confusionmatrix, wrong predictions may lead to fractional counts. For example,
    consider the gold-standard multi-label [A] and the predicted multi-label [B C] of size n=2.
    This leads to the increase of the cell {A,B} with 1/n=0.5 and also of the cell {A,C} with 
    1/n=0.5. This means that the normal count of 1 is distributed over all the labels in the
    multi-label.  
    
    Because sometimes, the number of labels is different in gold-standard and in the prediction,
    an empty label is introduced. In the example, the empty label gets a false negative count of
    1 because there was 1 label more in the prediction. If there had been a surplus of labels in
    the gold-standard, the empty label would get FP counts.
    
    The FP and FN counts of the empty label contain information on the different number of labels
    in a multi-label and, as such, they carry no meaning in a single label problem.
    
    
3. COMPUTED SCORES
    * Per label
    - precision (P) : TP/(TP+FP) for a given label
    - recall    (P) : TP/(TP+FN) for a given label
    - F-score   (P) : (1+beta^2)*TP/((1+beta^2)*TP+beta^2*FP+FN) for a given label
    
    * Macro-averaged
    sum(score_per_label)/N
    With N the number of labels in the training set or in the gold standard
    of the test set, depending on -t. The empty label is not counted.
        
    * Label-frequency-based micro-averaged
    sum( label_frequency*score_per_label )
    The label frequency can be retrieved from the training corpus (-t) or the test corpus. The 
    empty label is not included in the frequency counts. Labels that do not occur in the test corpus,
    but do occur in the training corpus are included when -t is set.
        
    * Micro-averaged
    score( sum(tp_per_label), sum(fp_per_label), sum(fn_per_label) )     
    Independent of option -t. The FP and FN of the empty label are included. 
        
    * Accuracy
    Accuracy is always computed by taking into account the entire (multi-)label.
        

4 . GENERAL RELATIONS
    * With the -m option set or when there are no multi-labels:
        MICRO-AVERAGED PRECISION always equals MICRO-AVERAGED RECALL.
        MICRO-AVERAGED F-SCORE is insensitive to beta.
    * If the labels are no multi-labels:
        ACCURACY always equals MICRO-AVERAGED F-SCORE.'
    * The empty label (*) never has a true positive.'
    
5. COMMAND LINE
    For the command line version, the macro-averaged and label-frequency-based micro-averaged scores
    are computed with the setting: strict=True, compute_none=False. The micro-averaged scores are
    computed with the settings: strict=False, compute_none=True.

    * compute_none

    The reason for this is that for label-frequency-based micro-averaged scores, it does not make sense
    to take into account the empty label during the frequency counting. The empty label will never have
    an influence because the precision and recall of the empy label are always zero. The macro-averaged
    is treated the same way for consistency.

    For the micro-averaged scores, it does make sense to take into account the empty labels because they
    capture information about errors on the different number of labels in the multi-labels.

    This is only an issue with multi-labels.

    * strict

    For macro-averaged and label-frequency-based micro-averaged scores, if no training is provided, it is
    better to use only gold-standard labels, since adding the predicted labels would disproportionally
    penalize the system that predicts erroneous labels that are not present in the gold-standard of test,
    but only in training, compared to a system that only makes erroneous predictions with labels from the
    test set.

    For micro-averaged scores, the predicted labels are needed to capture information about erroneous
    predictions with labels that are not in the test set.
    '''
          
              
def _usage():
    print >>sys.stderr, '''Compute evaluation statistics (version %s)
    
The script reports on:
    - true positive, false positive, and false negative counts for each label.
    - precision, recall, F-score, and accuracy for each label and averaged
      in different manners.
    - the confusionmatrix.
    
USAGE
    $ python confusionmatrix.py [-g index] [-p index] [-G index] [-f fsep] [-l lsep] [-b beta] [-m] [-t training_file] [-C cutoff] [-I pattern] [-T translation] [-L] [-F] [-M] [-v] [-V] test_file [prediction_file]
    
    test_file:       a file with a gold-standard/predicted label values.
                     One pair per line. The line should consist of a number
                     of fields (at least 2), separated by fsep. One field
                     should contain the gold-standard, another should contain
                     the predicted label. The other fields are free. 
              
                     This is useful for evaluating TiMBL output.
              
    OR
    
    test_file:       a file with the gold-standard. One value per line.
                     The line should consist of a number of fields (at least 1),
                     separated by fsep. One field should contain the gold-standard.
                     The other fields are free.
    prediction_file: One value per line. The lines should be paired with those
                     of test_file and should gold the predicted labels.
              
                     This is useful for evaluating e.g. SVM Light output. For SVM Light,
                     -C should be set also.


OPTIONS
    -g index: the index of the field containing the gold-standard.
              Starting at 0. (default: penultimate field)
    -p index: the index of the field containing the predicted label.
              Starting at 0. (default: last field)
    -G index: the index of the field containing the gold-standard in the optional training file.
              Starting at 0. (default: last field)
    
    -f fsep: the field separator (default: any whitespace)
    -l lsep: the label separator (default: _)
    
    -b beta: set the beta value for the F-scores (default: 1)
    -m: if set, the labels are considered to be combinations of multiple labels
        separated by lsep. 
        
    -t training_file: a file in the same format as test file. Is used to compute alternative
                      evaluation scores. See AVERAGE SCORES 
    
    -C cutoff: A float. When set, a prediction_file should be given also. When a prediction has a
               value below this cutoff, the label becomes -1; otherwise it becomes 1. Useful for
               evaluating SVM output.
    -I pattern: a regex pattern. Lines in the test_file that match this pattern are ignored.
                For example, to evaluate MBT files, set -I "<utt>".
    -T translation: A path to a file with the format:
                      old_label > new_label
                        ...
                    When provided, the labels as they are in training, prediction and test files are
                    translated into the new labels.
    -u label: the label that is given to SVM instances that have a value under cutoff -C (default: -1)
    -a label: the label that is given to SVM instances that have a value above cutoff -C (default: +1)
    
    -L: Suppress reporting in individual labels. Useful when there are a lot of labels.
    -F: Force reporting on micro-averaged precision and recall.
    -M: Don't print the confusionmatrix. Useful when there are a lot of different labels.
    -v: Print more information
    -V: Print information about the method of calculation
    
    
AVERAGE SCORES
    There are two ways to compute averaged scores: by using information from the training corpus
    or not. Option -t.
    
    For the macro-averaged scores, using only information from the test corpus may lead to
    different scores compared to using label information from the training corpus when
    there are labels in the training corpus that do not occur in the test corpus. Indeed, 
    the number of labels is used to average the scores.
    
    For the label-frequency-based micro-averaged scores, the relative frequencies of the
    labels are used. The frequencies may originate from the test or the training file.
    
    For micro-averaged scores, the training information has no influence. 
    
    - When comparing different test files that are labeled with the same training set, it
      is preferred to use option -t.
    - When comparing the same test file that is labeled using  different training files,
      it is preferred not to use option -t.
    - When the same test file is labeled with the same training file in different runs
      (for example because the machine learner had other settings), you should choose
       and stick to your choice.
    - When different test files are labeled with different training files, you should
      decide whether the test files or the training files are more similar in regard
      to relative label frequencies. If the training files are more similar, use -t, 
      otherwise don't.
    
    
EXAMPLES
    1. TiMBL
    +++++++++++
    To use on a TiMBL output file:
        $ timbl -f dimin.train -t dimin.test
        $ ./confusionmatrix.py -f ',' -t dimin.train dimin.test.IB1.O.gr.k1.out
    
    2. SVMLight (binary)
    +++++++++++++++++++++
    To use on SVMLight test and prediction file:
        $ ./svm_learn train_file model
        $ ./svm_classify test_file model prediction_file
        $ ./confusionmatrix.py -C 0 -g 0 test_file prediction_file
        
        or if you want to use the training file:
        $ ./confusionmatrix.py -C 0 -g 0 -t train_file test_file prediction_file
        
    SVM Light itself reports accuracy and precision/recall for label 1 when the training
    file is not used.
        
    Note that this only works for binary, non-multilabel SVMs. The class labels should be -1 or 1.
    The values below cutoff c, get label -1.
    
    Tested with SVM-light V6.02
    
ACKNOWLEDGEMENTS
    Grigorios Tsoumakas, Ioannis Katakis, and Ioannis Vlahavas, Mining Multi-label Data, Data Mining and Knowledge Discovery Handbook, Part 6, O. Maimon, L. Rokach (Ed.), Springer, 2nd edition, pp. 667-685, 2010.
    Daelemans, W., Zavrel, J., van der Sloot, K., & van den Bosch, A. (2010). Timbl: Tilburg memory-based learner, version 6.3. Tech. Rep. ILK 10-01, Tilburg University
    van Rijsbergen, C. J. (1975). Information Retrieval . London, UK: Butterworths.
    
SEE ALSO
    http://www.clips.ua.ac.be/~vincent/pdf/microaverage.pdf

%s, %s''' %(__version__, __date__, __author__)
    
    
    
    
if __name__ == '__main__':        
    try:
        opts,args=getopt.getopt(sys.argv[1:],'ht:mf:l:g:c:p:G:vMb:C:T:VNI:SLFu:a:', ['help'])
    except getopt.GetoptError:
        # print help information and exit:
        _usage()
        sys.exit(2)

    training_file=None
    multi=False
    verbose=False
    beta=1
    compute_none=False
    force=False
    
    fsep=None
    lsep='_'
    
    gold_index=-2
    pred_index =-1
    train_gold_index=-1
    
    print_cm=True
    print_more_info=False
    print_labels=True
    
    strict=False
    
    # for SVM
    prediction_file=None
    cutoff=None
    pos_label='+1'
    neg_label='-1'
    
    ignore=[]
    translation_file=None
    
    for o, a in opts:
        if o in ('-h', '--help'):
            _usage()
            sys.exit()
        if o == '-t':
            training_file = a
        if o == '-m':
            multi=True
        if o == '-f':
            fsep = a
        if o == '-l':
            lsep = a
        if o in ('-c', '-g'):
            # Added -c for backwards compatibility
            gold_index=int(a)
        if o == '-p':
            pred_index=int(a)
        if o == '-G':
            train_gold_index=int(a)
        if o == '-v':
            verbose=True
        if o == '-M':
            print_cm=False
        if o == '-b':
            beta=float(a)
        if o == '-C':
            cutoff=float(a)
        if o =='-T':
            translation_file=a
        if o == '-V':
            print_more_info=True
        if o == '-N':
            compute_none=True
        if o == '-I':
            ignore=[re.compile(a)]
        if o == '-S':
            strict=True
        if o == '-L':
            print_labels=False
        if o == '-F':
            force=True
        if o == '-u':
            neg_label=a
        if o == '-a':
            pos_label=a


    if not args and print_more_info:
        print_info()
        sys.exit(0)

    if len(args) not in [1,2]:
        _usage()
        sys.exit(1)
        
    if lsep == fsep:
        print >>sys.stderr, 'ERROR: fsep cannot be the same as lsep'
        sys.exit(1)
        
    test_file = args[0]

    if len(args) == 2: prediction_file = args[1]

    # RUN
    '''
    main(test_file, gold_index=gold_index, pred_index=pred_index, fsep=fsep, lsep=lsep, ignore=ignore, multi=multi, training_file=training_file, compute_none=compute_none,\
         train_gold_index=train_gold_index, beta=beta, print_cm=print_cm, verbose=verbose, cutoff=cutoff, prediction_file=prediction_file, translation_file=translation_file,\
         strict=strict, print_labels=print_labels)
'''
    meddling_main(test_file, gold_index=gold_index, pred_index=pred_index, fsep=fsep, lsep=lsep, ignore=ignore, multi=multi, training_file=training_file,\
         train_gold_index=train_gold_index, beta=beta, print_cm=print_cm, verbose=verbose, cutoff=cutoff, prediction_file=prediction_file, translation_file=translation_file,\
         print_labels=print_labels, force=force, pos_label=pos_label, neg_label=neg_label)

    if print_more_info: print_info()
        
