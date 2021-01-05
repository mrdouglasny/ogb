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
    return parser.parse_args(args)

args = parse_args()

data = np.load(args.infile)

if args.print_norms:
    with open(args.outfile, 'w') as out:
        print( 'norms:', np.linalg.norm(data, ord=1, axis=1).tolist(), file=out )
        motif = data[1,]+data[2,]-data[3,]
        print( 'motif:', np.linalg.norm(motif, ord=1), file=out )    
        if args.print_normsx and data.shape[0]>4:
            motif2 = data[4,]+data[5,]-data[6,]
            print( 'motif2:', np.linalg.norm(motif2, ord=1), file=out )    
else:
    np.savetxt(args.outfile, data, fmt='%.8e', header='Read from ' + args.infile)
