mlbMarkov
=========
A basic Markov chain code to compute the expectation vlaue of runs in an inning, given n bases and m outs. No outs on bases, and no taking an extra base!

requires 
* numpy

The basic usage is, e.g.,
python mlbMarkov_nm.py -nbases 3 -nouts 3 -p0 0.69 -p1 0.23 -p2 0.05 -p3 0.005 -p4 0.025

the probabilities are set with 
-p0 [probability for out]
-p1 [prob for a 1-base hit]
-p2 [prob for a 2-base hit]
etc...

if the inputs dont add up to 1, they get rescaled so that they do...