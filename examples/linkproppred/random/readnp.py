import numpy as np
import networkx as nx
import argparse
from torch import load
import os
import re
import csv

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Convert numpy saved data to text',
        usage='readnp.py [<file>] [-h | --help]'
    )

    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('-pn', '--print_norms', action='store_true')
    parser.add_argument('-px', '--print_normsx', action='store_true')
    parser.add_argument('-P', '--PairRE', action='store_true')
    parser.add_argument('-t', '--tensor', action='store_true')
    return parser.parse_args(args)

def motif(v):
    if args.PairRE:
        d = v.shape[1]
        h = v[:,0:(d//2)]
        t = v[:,(d//2):d]
        return h[0,:]*h[1,:]*t[2,:] - t[0,:]*t[1,:]*h[2,:]
    else:
        return v[0,:]+v[1,:]-v[2,:]

args = parse_args()

data = np.load(args.infile)

if args.print_norms:
    with open(args.outfile, 'w') as out:
        print( 'norms:', np.linalg.norm(data, ord=1, axis=1).tolist(), file=out )
        print( 'motif:', np.linalg.norm(motif(data[1:4,:]), ord=1), file=out )    
        if args.print_normsx and data.shape[0]>4:
            print( 'motif2:', np.linalg.norm(motif(data[4:7,:]), ord=1), file=out )    
#        if args.print_pair:

elif args.tensor:
    s = np.array2string( data, format='%.8e', threshold=numpy.inf, suppress_small=true )
    with open(args.outfile, 'w') as out:
        print( s, file=out )
else:
    np.savetxt(args.outfile, data, fmt='%.8e', header='Read from ' + args.infile)
