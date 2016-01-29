
import numpy as np

import os, sys
import re
#import pickle
import numpy as np
import copy
#from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import seaborn as sns

class mlbMarkov:
    def __init__(self, vbose=0, nbases=3, nouts=3):
        self.max_score = 10
        self.vbose = vbose
        self.nbases = nbases
        self.nouts = nouts
        self.state2int = {}
        self.int2state = {}
        self.initEnumerateStates()
        self.sz = len(self.allStates)
        self.initTransitionMatrix()
        self.probs = {}
        self.probs = {}
        for i in range(10):
            self.probs[i] = 0
        self.init_probs()
        self.solvedSystem = None

        self.transitionMatrix = self.initTransitionMatrix()
        self.valueMatrix = self.initValueMatrix()
        self.runsMatrix = np.zeros((self.max_score+1, self.sz))
        self.makeTransitionMatrix()
        self.makeValueMatrix()
        self.summary_keys = ['man1', 'man2', 'man3', 'out0', 'out1', 'out2', 'out3']
        for i in range(self.max_score+1):
            self.summary_keys.append('run%d' % i)

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
        if newo>oldo or newo==3:
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
                inew = self.state2int[newState]
                self.valueMatrix[inew][iold] = self.getValue(oldState, newState)
                if self.vbose>=1:
                    print '** makeVM *******'
                    print iold, inew, self.valueMatrix[inew][iold]


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

    def parse_state(self, state, n=0):
        m3, m2, m1 = state.split('_')[0][:]
        nouts = int(state.split('_')[1][:])
        m3 = int(m3)
        m2 = int(m2)
        m1 = int(m1)
        nbase = m3+m2+m1
        nruns = n - nbase - nouts
        return {'man3': m3, 'man2': m2, 'man1': m1, 'nouts': nouts, 'nruns': nruns}

    def state_vector_to_summary(self, stateVector, n=0):
        data = {}
        for k in self.summary_keys:
            data[k] = 0

        print data
        for i, p in enumerate(stateVector):
            s = self.int2state[i]
            ans = self.parse_state(s, n=n)
            print s, ans, p
            if p==0:
                continue
            for k in ['man1', 'man2', 'man3']:
                data[k] += p*ans[k]
            assert ans['nruns']>=0 or p<1e-6
            k = 'run%d' % ans['nruns']
            data[k] += p
            k = 'out%d' % ans['nouts']
            data[k] += p

        return data

    def generate_sequence(self, v0, nseq=10):
        seq = []

        for i in range(nseq):
            if i==0:
                v = copy.copy(v0)
            else:
                v = self.transitionMatrix.dot(v)

            summary = self.state_vector_to_summary(v, n=i)
            print summary
            seq.append(summary)
        return seq

    def bar_chart_from_summary(self, summary, n=0, ysc='linear'):
        xx = []
        yy = []
        plt.clf()
        for i, k in enumerate(self.summary_keys):
            xx.append(i)
            yy.append(summary[k])

        plt.bar(np.array(xx)-0.3, yy, color='steelblue')
        plt.xticks(xx, self.summary_keys, rotation='vertical', fontsize=10)
        plt.yscale(ysc)
        plt.ylim(0.01,1)
        plt.xlim(-0.5, self.max_score+8)
        plt.text(self.max_score+7, 0.9, 'PA= %03d' % n, ha='right')
        plt.ylabel('probability')
        plt.title('baseball markov chain')
    def make_chart_from_state(self):
        x = np.zeros(3+3+(self.max_score+1))
        xx = range(len(x))
        plt.bar(xx,xx)


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

