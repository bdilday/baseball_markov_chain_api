#!/usr/bin/env python

import os, sys
import re
#import pickle
import numpy as np
#import copy
#from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

###################
class mlbMarkov:
    def __init__(self, nbases=3, nouts=3, vbose=0):
        self.vbose = vbose
        self.nbases = nbases
        self.nouts = nouts
        self.state2int = {}
        self.int2state = {}
        self.initEnumerateStates()
        self.initTransitionMatrix()
        self.probs = {}
        self.transitionMatrix = []
        self.valueMatrix = []
        self.probs = {}
        for i in range(10):
            self.probs[i] = 0
        self.solvedSystem = None

    def reNorm(self, a, norm=1.0):
        sum = 0.0
        for k in a:
            sum += a[k]

        v = norm/sum

        for k in a:
            a[k] *= v 

        return a

    def initEnumerateStates(self):
        nb = self.nbases
        nstate = 0
        for i in range(2**(self.nbases)):
            s = bin(i).split('b')[1]
            for j in range(self.nbases-len(s)+1-1):
                s = '0'+s
            for o in range(self.nouts):
                k = s + '_%02d' % o
                if self.vbose>=1:
                    print nstate, i, s, k
                self.state2int[k] = nstate
                self.int2state[nstate] = k
                nstate += 1

        allStates = self.state2int.keys()
        allStates.sort()
        self.allStates = allStates

    def getNewState(self, nbaseHit, oldState):
        nb, no = self.stateToInfo(oldState)
        if self.vbose>=1:
            print oldState, nbaseHit, nb, no
        # a new state comes from ,
        # multiply by 2 nb times. 
        # dont forget to add 1 the first time
        if nbaseHit == 0:
            return self.infoToState(nb, no+1)

        newNb = nb
        for i in range(nbaseHit):            

            newNb *= 2
            if i==0:
                newNb += 1
            if self.vbose>=1:
                print i, nb, newNb
        newNb = newNb % (2**(self.nbases))

        return self.infoToState(newNb, no)

    def infoToState(self, nb, no):

        s = bin(nb).split('b')[1]
        if self.vbose>=2:
            print nb, no, s, len(s), nb-len(s)+1
        for j in range(self.nbases-len(s)+1-1):
            s = '0'+s
        k = s + '_%02d' % no
        if self.vbose>=2:
            print 'infoToState', nb, no, k
        return k

    def stateToInfo(self, s):
        bb, oo = s.split('_')
        ii = int(bb, base=2)

        return ii, int(oo)

    def getNOnBase(self, state):
        s = state.split('_')[0]
        if self.vbose>=1:
            print 's', s
        sum = 0
        for i in s:
            sum += int(i)
            if self.vbose>=1:
                print 'sum', i, sum
        return sum

    def getValue(self, oldState, newState):
        oldb, oldo = self.stateToInfo(oldState)
        newb, newo = self.stateToInfo(newState)
        if newo>oldo:
            return 0
        # value is number that scored
        # this is, n_start + 1 = n_end + n_score
        # nscore = n_start + 1 - n_end
        n_start = self.getNOnBase(oldState)
        n_end = self.getNOnBase(newState)
        n_score = n_start+1-n_end
        return n_score

    def initTransitionMatrix(self):
        sz = len(self.int2state)
#        del self.transitionMatrix
        self.transitionMatrix = np.zeros((sz, sz))

    def initValueMatrix(self):
        sz = len(self.int2state)
#        del self.transitionMatrix
        self.valueMatrix = np.zeros((sz, sz))

    def makeTransitionMatrix(self, vbose=None):

        self.initTransitionMatrix()
        allStates = self.allStates

        for i, oldState in enumerate(allStates):
            # now, for each prob, we compute the prob to transition to new state
            iold = self.state2int[oldState]
            for nb in range(self.nbases+2):
                if self.vbose>=1:
                    print '** makeTM *******'
                if not nb in self.probs:
                    self.probs[nb] = 0
                v = self.probs[nb]
                newState = self.getNewState(nb, oldState)
                if not newState in self.allStates:
                    if self.vbose>=1:
                        print 'makeTM', oldState, nb, newState, iold, 'xxx'
                    continue
#                newB, newO = self.stateToInfo(
                inew = self.state2int[newState]
                self.transitionMatrix[inew][iold] += v
                if self.vbose>=1 or vbose>=1:
                    print 'makeTM', oldState, nb, newState, iold, inew, self.transitionMatrix[inew][iold]

    def makeValueMatrix(self):

        self.initValueMatrix()
        allStates = self.allStates

        for i, oldState in enumerate(allStates):
            iold = self.state2int[oldState]
            for j, newState in enumerate(allStates):
                if self.vbose>=1:
                    print '** makeVM *******'
                inew = self.state2int[newState]
                self.valueMatrix[inew][iold] = self.getValue(oldState, newState)

    def printSolution(self, printProbs=True):
        allStates = self.allStates
        if printProbs:
            for k in range(m.nbases+2):
                print 'prob', k, m.probs[k]
        print 'idx state expectRunsPerInn'
        for i, v in enumerate(self.solvedSystem):
            s = self.int2state[i]
            print '%3d %s %.3f' % (i, s, v)

    def printTransitionMatrix(self):
        print 'tm', np.shape(self.transitionMatrix)
        allStates = self.allStates
        for i, oldState in enumerate(allStates):
            iold = self.state2int[oldState]
            for j, newState in enumerate(allStates):
                inew = self.state2int[newState]
                print 'tm', iold, oldState, inew, newState, self.transitionMatrix[inew][iold]

    def printValueMatrix(self):
        print 'vm', np.shape(self.valueMatrix)
        allStates = self.allStates
        for i, oldState in enumerate(allStates):
            iold = self.state2int[oldState]
            for j, newState in enumerate(allStates):
                inew = self.state2int[newState]
                print 'vm', iold, oldState, inew, newState, self.valueMatrix[inew][iold]

    def solveSystem(self):
        nrow, ncol = np.shape(self.transitionMatrix)
        if not nrow==ncol:
            raise Exception

        m1 = np.identity(nrow)
        mt = self.transitionMatrix
        mv = self.valueMatrix
        vr = np.diag(np.dot(np.transpose(mv),mt))
        ans = np.linalg.solve(np.transpose(m1-mt), vr)
        self.solvedSystem = ans
        return ans



def loopAndPlot(nbasemin=3, nbasemax=3, noutmin=1, noutmax=20, probs=None, vbose=0, lclf=False, normIndex=-1):
    if lclf:
        plt.clf()
    for nb in range(nbasemin, nbasemax+1):
        xx = []
        yy = []
        for no in range(noutmin, noutmax+1):
            m = main(nbases=nb, nouts=no, vbose=vbose, probs=probs)
            m.makeTransitionMatrix()    
            m.makeValueMatrix()
            ans = m.solveSystem()
            print 'loop', nb, no, ans[0]
            yy.append(ans[0])
            xx.append(no)
        xx = np.array(xx)
        yy = np.array(yy)
        if normIndex>0:
            yy = yy/yy[normIndex]
        plt.plot(xx, yy, 'o-', label='%d bases' % nb)
    plt.grid(b=True)
    plt.legend(loc=2, fontsize='x-small')
    plt.xlabel('nouts')
    if normIndex>0:
        plt.ylabel('runs per inning / run per inning (%d outs)' % (normIndex+1))
    else:
        plt.ylabel('runs per inning')
    return xx, yy

def main(nbases=3, nouts=3, vbose=0, probs=None):
    m = mlbMarkov(nbases=nbases, nouts=nouts, vbose=vbose)

    for k in probs:
        m.probs[k] = probs[k]

    m.probs = m.reNorm(m.probs, norm=1.0)
    return m

###################
if __name__=='__main__':
    printProbs = True
    vbose = 0
    nbases = 3
    nouts = 3
    probs = {}
    probs[1] = 0.15+0.08
    probs[2] = 0.05 
    probs[3] = 0.005 
    probs[4] = 0.025
    probs[0] = 1-(probs[1]+probs[2]+probs[3]+probs[4])

    for ia, a in enumerate(sys.argv):
        if a=='-nbases' or a=='-nbase':
            nbases = int(sys.argv[ia+1])
        if a=='-nouts' or a=='-nout':
            nouts = int(sys.argv[ia+1])
        if a[0:2]=='-p':
            m = re.search('-p([0-9]+)', a)
            ib = int(m.group(1))
            v= float(sys.argv[ia+1])
            probs[ib] = v
        if a=='-vbose':
            vbose = int(sys.argv[ia+1])

        if a=='-printProb' or a=='-printProbs':
            printProbs = bool(int(sys.argv[ia+1]))

    
    m = main(nbases=nbases, nouts=nouts, vbose=vbose, probs=probs)
    m.probs = m.reNorm(m.probs)
    m.makeTransitionMatrix()    
    m.makeValueMatrix()

    # test
    if vbose>=1:
        print '***************************'
    for oldState in m.allStates:
        if vbose>=1:
            print '***************************'
        for nb in range(m.nbases+2):
            if vbose>=1:
                print '*****************'
            newState = m.getNewState(nb, oldState)
            if not newState in m.allStates:
                continue
            nonO = m.getNOnBase(oldState)
            nonN = m.getNOnBase(newState)
            val = m.getValue(oldState, newState)
            if vbose>=1:
                print 'xxx', oldState, nonO, nb, newState, nonN, val, 'xxx'


    ans = m.solveSystem()
    m.printSolution(printProbs=printProbs)
