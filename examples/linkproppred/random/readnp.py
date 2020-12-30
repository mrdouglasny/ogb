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
    return parser.parse_args(args)

args = parse_args()

data = np.load(args.infile)

if args.print_norms:
    print( 'norms:', np.linalg.norm(data, ord=1, axis=0) )
    motif = data[0,]+data[1,]-data[2,]
    print( 'motif:', np.linalg.norm(motif, ord=1) )    
else:
    np.savetxt(args.outfile, data, fmt='%.8e', header='Read from ' + args.infile)
