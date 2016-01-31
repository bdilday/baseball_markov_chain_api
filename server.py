
from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
from flask.ext.cors import CORS
import json
import re

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
                                             max_score=a['max_score'],
                                             nbases=a['nbases'],
                                             nouts=a['nouts'])
    ans = mk.server_hook(nseq=a['nseq'])
    return json.dumps(ans)

def validate_request(request):
    a = request.args
    ans = {}
    ans['nbases'] = min(int(a.get('nbases', 3)), 6)
    ans['nouts'] = min(int(a.get('nouts', 3)), 10)
    ans['nseq'] = min(int(a.get('seq_length', 10)), 100)
    ans['probs'] = {}
    for k in a.keys():
        m = re.match('^p([0-9]+)$', k)
        if m:
            idx = int(m.group(1))
            ans['probs'][idx] = float(a.get(k))
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
