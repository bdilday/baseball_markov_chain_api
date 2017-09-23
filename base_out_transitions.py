
import sys
import re
import json
import baseball_markov_chain_api


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
    ofile = 'a.json'

    for ia, a in enumerate(sys.argv):
        if a=='-nbases' or a=='-nbase':
            nbases = int(sys.argv[ia+1])
        if a=='-nouts' or a=='-nout':
            nouts = int(sys.argv[ia+1])
        if a == '-ofile':
            ofile = sys.argv[ia + 1]

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

    m = baseball_markov_chain_api.main(nbases=nbases,
                                       nouts=nouts,
                                       vbose=vbose, probs=probs, max_score=0)
    m.probs = m.reNorm(m.probs)
    m.makeTransitionMatrix()
    m.makeValueMatrix()
    ans = m.generate_detailed_sequence(m.v0)
    ans = [a for a in ans if a['p_end']>0]

    with open(ofile, 'w') as fh:
        json.dump(ans, fh)
