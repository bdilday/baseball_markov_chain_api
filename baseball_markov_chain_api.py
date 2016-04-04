
import sys
import re
import numpy as np
import copy

class mlbMarkov:
    def __init__(self, vbose=0, nbases=3, nouts=3, max_score=15, probs=None):
        self.n_in_lineup = 9
        self.innings = 9
        self.max_score = max_score
        self.vbose = vbose
        self.nbases = nbases
        self.nouts = nouts
        self.total_outs = self.innings*self.nouts
        self.state2int = {}
        self.int2state = {}
        self.initEnumerateStates()
        self.sz = len(self.allStates)
        self.initTransitionMatrix()
        self.probs = {}

        self.lineup_probs = self.set_lineup()

        self.init_probs()

        if probs is not None:
            for k, v in probs.items():
                self.probs[k] = v

        blacklist = []
        for k in self.probs:
            if k>(self.nbases+1):
                blacklist.append(k)
        for k in blacklist:
            self.probs.pop(k)

        self.probs = self.reNorm(self.probs)
        assert abs(sum([float(v) for v in self.probs.values()])-1)<1e-6, self.probs
        self.probs_dict = {}
        for i in range(self.nbases+2):
            self.probs_dict['prob%d' % i] = self.probs[i]

        self.solvedSystem = None

        self.transitionMatrix = self.initTransitionMatrix()
        self.valueMatrix = self.initValueMatrix()
        self.runsMatrix = np.zeros((self.max_score+1, self.sz))
        self.makeTransitionMatrix()
        assert np.all(abs(self.transitionMatrix.sum(0)-1)<1e-6), self.transitionMatrix.sum(0)

        self.summary_keys = ['out0']
        for i in range(self.nbases):
            self.summary_keys.append('man%d' % (i+1))
        for i in range(self.total_outs):
            self.summary_keys.append('out%d' % (i+1))

        for i in range(self.max_score+1):
            self.summary_keys.append('run%d' % i)
        self.v0 = np.zeros((self.sz, 1))
        self.v0[0] = 1

    def set_lineup(self, cfgfile='config.txt'):
        # TODO: generalize to more than 3 bases
        aa = {}
        lines = [l.strip() for l in open(cfgfile).readlines() if len(l)>0]
        for l in lines:
            lineup_idx, h1, h2, h3, h4 = l.split()
            lidx = int(lineup_idx)
            aa[lidx] = {}
            aa[lidx][0] = 1.0
            for i, v in enumerate([h1, h2, h3, h4]):
                aa[lidx][i+1] = float(v)
                aa[lidx][0] -= float(v)
            assert abs(sum([float(v) for v in aa[lidx].values()])-1)<1e-6, aa
        return aa

    def init_probs(self):
        if self.nbases==3:
            self.probs[1] = 0.15+0.08
            self.probs[2] = 0.05
            self.probs[3] = 0.005
            self.probs[4] = 0.025
            self.probs[0] = 1-(self.probs[1]+self.probs[2]+self.probs[3]+self.probs[4])
        else:
            v = 1.0/(self.nbases+2)
            for i in range(self.nbases+2):
                self.probs[i] = v

    def reNorm(self, a, norm=1.0):
        sum = 0.0
        for k in a:
            sum += a[k]
        v = norm/sum
        for k in a:
            a[k] *= v

        return a

    def bases_string(self, i):
        s = bin(i).split('b')[1]
        for j in range(self.nbases-len(s)+1-1):
            s = '0'+s
        return s

    def initEnumerateStates(self):
        nstate = 0
        for i in range(2**(self.nbases)):
            s = self.bases_string(i)
            for o in range(self.total_outs+1):
                for r in range(self.max_score+1):
                    k = s + '_%02d_%02d' % (o, r)
                    if self.vbose>=1:
                        print nstate, i, s, k
                    self.state2int[k] = nstate
                    self.int2state[nstate] = k
                    nstate += 1

        allStates = self.state2int.keys()
        allStates.sort()
        self.allStates = allStates

    def getNewState(self, nbaseHit, oldState):
        nb, no, rr = self.stateToInfo(oldState)
        old_inning_outs = no % self.nouts

        if self.vbose>=1:
            print 'old nb', oldState, nbaseHit, nb, no

        if no==self.total_outs:
            # cant transition if all outs already used up
            return oldState

        # it was an out
        if nbaseHit == 0:
            # clear bases if it was the last out of the inning
            nb = 0 if ((no+1) % self.nouts) == 0 else nb
            return self.infoToState(nb, no+1, rr)

        # a new state comes from ,
        # multiply by 2 nb times.
        # dont forget to add 1 the first time
        newNb = nb
        for i in range(nbaseHit):

            newNb *= 2
            if i==0:
                newNb += 1
            if self.vbose>=1:
                print i, nb, newNb

        newNb = newNb % (2**(self.nbases))
        newState = self.infoToState(newNb, no)
        rr = int(rr) + self.getValue(oldState, newState)
        rr = min(rr, self.max_score)
        return self.infoToState(newNb, no, rr)

    def infoToState(self, nb, no, rr=None):

        s = bin(nb).split('b')[1]
        if self.vbose>=2:
            print nb, no, s, len(s), nb-len(s)+1
        for j in range(self.nbases-len(s)+1-1):
            s = '0'+s

        k = s + '_%02d' % no
        if rr is not None:
            k += '_%02d' % rr
        if self.vbose>=2:
            print 'infoToState---------', nb, no, k
        return k

    def stateToInfo(self, s):
        assert s.count('_') in [1,2], s
        if s.count('_')==2:
            bb, oo, rr = s.split('_')
            rr = int(rr)
        else:
            bb, oo = s.split('_')
            rr = None
        ii = int(bb, base=2)

        return ii, int(oo), rr

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
        oldb, oldo, oldr = self.stateToInfo(oldState)
        newb, newo, newr = self.stateToInfo(newState)
        new_inning_outs = newo % self.nouts
        if newo>oldo or new_inning_outs==self.nouts:
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
        self.transitionMatrix = np.zeros((sz, sz))

    def initValueMatrix(self):
        sz = len(self.int2state)
        self.valueMatrix = np.zeros((sz, sz))

    def makeTransitionMatrix(self, vbose=None):
        self.initTransitionMatrix()
        allStates = self.allStates

        for i, oldState in enumerate(allStates):
            # now, for each prob, we compute the prob to transition to new state
            assert oldState in allStates
            iold = self.state2int[oldState]

            oldb, oldo, oldr = self.stateToInfo(oldState)

            for nb in range(self.nbases+2):
                if self.vbose>=1:
                    print '** makeTM *******'
                assert nb in self.probs, nb

                lineup_idx = (oldb + oldo + oldr + 1) % self.n_in_lineup
                lineup_idx = 9 if lineup_idx==0 else lineup_idx
                assert lineup_idx>=1 and lineup_idx<=9, lineup_idx

                if lineup_idx in self.lineup_probs:
                    v = self.lineup_probs[lineup_idx][nb]
                else:
                    # fall back to default
                    v = self.probs[nb]

                newState = self.getNewState(nb, oldState)
                assert newState in allStates, newState
                if not newState in self.allStates:
                    if self.vbose>=1:
                        print 'makeTM', oldState, nb, newState, iold, 'xxx'
                    continue
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
        ans = {}
        bs = list(state.split('_')[0][:])
        counter = 1
        while len(bs)>0:
            x = int(bs.pop())
            k = 'man%d' % counter
            ans[k] = x
            counter += 1

        nouts = int(state.split('_')[1][:])
        nruns = int(state.split('_')[2][:])
        ans['nruns'] = nruns
        ans['nouts'] = nouts
        return ans

    def state_vector_to_summary(self, stateVector, n=0):

        data = {}
        for k in self.summary_keys:
            data[k] = 0

        mean_runs = 0.0
        if self.vbose>=1:
            print data
        for i, p in enumerate(stateVector):
            s = self.int2state[i]
            ans = self.parse_state(s)
            if self.vbose>=1:
                print s, ans, p
            if p[0]==0:
                continue
            for ibase in range(self.nbases):
                k = 'man%d' % (ibase+1)
                data[k] += p[0]*ans[k]
            assert ans['nruns']>=0 or p<1e-6
            k = 'run%d' % ans['nruns']
            data[k] += p[0]
            mean_runs += ans['nruns']*p[0]
            k = 'out%d' % ans['nouts']
            data[k] += p[0]
        data['mean_runs'] = mean_runs
        return data

    def generate_sequence(self, v0, nseq=10):
        seq = []

        for i in range(nseq):
            if i==0:
                v = copy.copy(v0)
            else:
                v = self.transitionMatrix.dot(v)

            summary = self.state_vector_to_summary(v, n=i)
            if self.vbose>=1:
                print summary
            seq.append(summary)
        return seq

    def transitionMatrixOutputArray(self, threshold=1e-6):
        ans = []
        for j in range(self.sz):
            for i in range(self.sz):
                p = self.transitionMatrix[j][i]
                if p>threshold:
                    ans.append((i, self.int2state[i],
                               j, self.int2state[j],
                               self.transitionMatrix[j][i]
                               ))
        return ans

    def printTransitionMatrix(self, threshold=1e-6):
        ans = self.transitionMatrixOutputArray(threshold=threshold)
        for line in ans:
            print line

    def printStateVector(self, v, threshold=1e-6):
        assert len(v)==self.sz
        s = 0.0
        counter = 0
        maxp = -1
        maxi = None
        for i, p in enumerate(v):
            if p<threshold:
                continue
            print '%03d %s %.6f ' % (i, self.int2state[i], p[0])
            s += p[0]
            counter += 1
            if p[0]>maxp:
                maxp = p[0]
                maxi = i

        assert maxi is not None
        print '---------------------'
        descrpition = '%d of %d' % (counter, self.sz)
        print '%13s %.6f' % (descrpition, s)
        print 'max %s %.4f' % (self.int2state[maxi], maxp)

    def server_hook(self, **kwargs):
        nseq = kwargs['nseq']
        seq = self.generate_sequence(self.v0, nseq=kwargs['nseq'])

        return {'seq': seq, 'probs': self.probs_dict, 'seq_length': nseq}

    def average_runs_from_state_vector(self, v):
        a = 0.0
        for i, p in enumerate(v):
            _, __, rr = self.stateToInfo(self.int2state[i])
            a += rr*p
        return a

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
    m.generate_sequence(m.v0)
