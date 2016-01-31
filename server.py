
from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
from flask.ext.cors import CORS
import json

import baseball_markov_chain_api

server = Flask(__name__)
CORS(server)

@server.route('/')
def index():
    return render_template('index.html')

@server.route('/baseball_markov_chain', methods=['GET'])
def baseball_markov_chain_endpoint():
    a = validate_request(request)
    probs = reNorm(a['probs'])
    mk = baseball_markov_chain_api.mlbMarkov(vbose=0,
                                             probs=probs,
                                             max_score=a['max_score'])
    ans = mk.server_hook(nseq=a['nseq'])
    return json.dumps(ans)

def validate_request(request):
    a = request.args
    ans = {}
    ans['nseq'] = min(int(a.get('seq_length', 10)), 100)
    ans['p0'] = float(a.get('p0', 0.69))
    ans['p1'] = float(a.get('p1', 0.23))
    ans['p2'] = float(a.get('p2', 0.05))
    ans['p3'] = float(a.get('p3', 0.005))
    ans['p4'] = float(a.get('p4', 0.025))
    ans['probs'] = {'p0': ans['p0'], 'p1': ans['p1'], 'p2': ans['p2'],
                    'p3': ans['p3'], 'p4': ans['p4']}
    ans['max_score'] = min(int(a.get('max_score', 10)), 30)
    return ans

def reNorm(a, norm=1.0):
    sum = 0.0
    for k in a:
        sum += a[k]
    v = norm/sum
    for k in a:
        a[k] *= v
    return a

if __name__ == '__main__':
    server.run(host='127.0.0.1', port=8001, debug=True)
