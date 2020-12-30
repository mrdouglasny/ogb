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
    return parser.parse_args(args)

args = parse_args()

data = np.load(args.infile)
np.savetxt(args.outfile, data, fmt='%.8e')
