
import numpy as np

import os, sys
import re
#import pickle
import numpy as np
import copy
#from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

class mlbMarkov:
    def __init__(self, vbose=0, nbases=3, nouts=3):

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
        self.init_probs()
        self.solvedSystem = None

    def init_probs(self):
        self.probs[1] = 0.15+0.08
        self.probs[2] = 0.05
        self.probs[3] = 0.005
        self.probs[4] = 0.025
        self.probs[0] = 1-(self.probs[1]+self.probs[2]+self.probs[3]+self.probs[4])


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

        # the 3 out state
        k = '000_03'
        self.state2int[k] = nstate
        self.int2state[nstate] = k
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

        if int(no) == 3:
            s = '000'

        k = s + '_%02d' % no
        if self.vbose>=2:
            print 'infoToState---------', nb, no, k
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
        last = self.transitionMatrix.shape[0]-1
        self.transitionMatrix[last][last] = 1

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

    def printSolution(self, state_vector, printProbs=True):
        print 'idx state prob'
        for i, v in enumerate(state_vector):
            s = self.int2state[i]
            print '%3d %s %.6f' % (i, s, v)


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
    ninn = 9

    probs = {}
    probs[1] = 0.15+0.08
    probs[2] = 0.05
    probs[3] = 0.005
    probs[4] = 0.025
    probs[0] = 1-(probs[1]+probs[2]+probs[3]+probs[4])
    doLinearWeights = False

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

        if a=='-doLinearWeights':
            doLinearWeights = bool(int(sys.argv[ia+1]))

    for i in range(nbases+2):
        if not i in probs:
            probs[i] = 0

    m = main(nbases=nbases, nouts=nouts, vbose=vbose, probs=probs)
    m.probs = m.reNorm(m.probs)
    m.makeTransitionMatrix()
    m.makeValueMatrix()

