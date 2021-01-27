from ogb.io import DatasetSaver
from ogb.linkproppred import LinkPropPredDataset
import numpy as np
import networkx as nx
import argparse
from torch import load
import os
import re
import csv
import sys

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Convert Knowledge Graph Formats',
        usage='convertkg.py [<args>] [-h | --help]'
    )

    parser.add_argument('dataset', type=str)
    parser.add_argument('-t', '--do_test', action='store_true')
    parser.add_argument('--print_relations', action='store_true')
    parser.add_argument('--select_head', type=int, default=-1)
    parser.add_argument('--select_tail', type=int, default=-1)
    parser.add_argument('--test_upto', type=int, default=0)
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-m', '--mode', type=str, help='read_triples,read_two_files,random_gnp')
    parser.add_argument('-s', '--subsample', type=float, default=0.8)
    parser.add_argument('--shuffle_edge_types', type=float, default=0.0)
    parser.add_argument('--collapse_edge_types', type=int, default=0, help='reduce edge types to n mod N')
    parser.add_argument('-ep', '--edge_probability', type=float, default=0.1)
    parser.add_argument('-nv', '--n_vertices', type=int)
    parser.add_argument('-nr', '--n_relations', type=int)
    parser.add_argument('--map_node_file', type=str)
    parser.add_argument('--map_relation_file', type=str)
    return parser.parse_args(args)

def read_map(file):
    if file==None:
        return None
    map = dict()
    with open(file, newline='') as csvfile:
        csvfile.readline()
        for (idx,name) in csv.reader(csvfile, delimiter=',', quotechar='|'):
            map[name] = int(idx)
    return map

args = parse_args()
dataset_name = args.dataset

if args.do_test:
    meta = 'dataset_' + re.sub('-','_',args.dataset) + '/meta_dict.pt'
    meta_dict = load(meta)
    dataset = LinkPropPredDataset(dataset_name, meta_dict = meta_dict)
    dsplit = dataset.get_edge_split()
    if args.print_relations:
        np.set_printoptions(threshold=np.inf)
        print('test.relations <- c(')
        print(re.sub('[\[\]]', '', np.array2string( dsplit['test']['relation'], separator=', ' )))
        print(')')
    elif args.select_head>=0 or args.select_tail>=0:
        for k in dsplit.keys():
            for i in range(len(dsplit[k]['head'])):
                (h,t,r) = (dsplit[k]['head'][i],dsplit[k]['tail'][i],dsplit[k]['relation'][i])
                if args.select_head<0 or args.select_head==h:
                    if args.select_tail<0 or args.select_tail==t:
                        print( k, '(', h, ',', r, ',', t, ')' )
    else:
        print(dataset[0])
        print(dsplit)
    exit(0)

num_vertices = args.n_vertices
num_relations = args.n_relations
graph = dict()

node_map = read_map(args.map_node_file)
relation_map = read_map(args.map_relation_file)
if node_map!=None and relation_map==None:
    raise ValueError('need map_relation_file')

    
if args.mode=='random_gnp':
# generate a random graph in the object 'graph'
    g = nx.fast_gnp_random_graph(num_vertices, args.edge_probability)
    graph['edge_index'] = np.array(g.edges).transpose() 
    graph['num_nodes'] = len(g.nodes)
    num_edges = graph['edge_index'].shape[1]
    graph['edge_reltype'] = np.random.randint(num_relations, size=(num_edges,1)) 
elif args.mode=='read_triples':
    edges = []
    relations = []
    num_nodes = 0
    with open(args.file, newline='') as csvfile:
        for row in csv.reader(csvfile, delimiter=',', quotechar='|'):
            if node_map!=None:
                (head,relation,tail) = (node_map[row[0]],relation_map[row[1]],node_map[row[2]])
            else:
                (head,relation,tail) = [int(r) for r in row]
            edges.append((head,tail))
            relations.append(relation)
            if head>=num_nodes:
                num_nodes = head+1
            if tail>=num_nodes:
                num_nodes = tail+1
    graph['edge_index'] = np.array(edges).transpose()
    graph['num_nodes'] = num_nodes
    graph['edge_reltype'] = np.reshape( np.array(relations), (-1,1) )
elif args.mode=='read_two_files':
    edges = []
    relations = []
    num_nodes = 0
    with open(args.file+'/edge.csv', newline='') as edge_file:
        for row in csv.reader(edge_file, delimiter=',', quotechar='|'):
            (head,tail) = [int(r) for r in row]
            edges.append((head,tail))
            if head>=num_nodes:
                num_nodes = head+1
            if tail>=num_nodes:
                num_nodes = tail+1
    with open(args.file+'/edge_reltype.csv', newline='') as rel_file:
        for row in csv.reader(rel_file, delimiter=',', quotechar='|'):
            relations.append(int(row[0]))
    graph['edge_index'] = np.array(edges).transpose()
    graph['num_nodes'] = num_nodes
    graph['edge_reltype'] = np.reshape( np.array(relations), (-1,1) )
else:
    raise ValueError('unknown mode')

# print(graph)

#### should not need to modify below this line

num_edges = graph['edge_index'].shape[1]
old_edge_reltype = np.copy( graph['edge_reltype'] )

if args.shuffle_edge_types>0.0:
    print( 'old edge_types', graph['edge_reltype'][:,0] )
    select = np.random.sample(size=num_edges) < args.shuffle_edge_types
    sh_types = np.extract( select, graph['edge_reltype'][:,0] )
    np.random.shuffle( sh_types )
    np.putmask( graph['edge_reltype'][:,0], select, sh_types )
    print( 'shuffle', len(sh_types), 'out of', num_edges )
    print( 'new', graph['edge_reltype'][:,0] )

if args.collapse_edge_types>0:
    N = args.collapse_edge_types
    for i in range(num_edges):
        graph['edge_reltype'][i,0] = graph['edge_reltype'][i,0] % N
    print( 'new', graph['edge_reltype'][:,0] )

n_changed = 0
for i in range(num_edges):
    if old_edge_reltype[i,0] != graph['edge_reltype'][i,0]:
        n_changed += 1
print ('n_changed =', n_changed, '/', num_edges )

def make_triples( graph, idx ):
    triples = dict()
    triples['head'] = graph['edge_index'][0,idx]
    triples['tail'] = graph['edge_index'][1,idx]
    triples['relation'] = graph['edge_reltype'][idx,0] 
    return triples

# constructor
saver = DatasetSaver(dataset_name = dataset_name, is_hetero = False, version = 1, root='dataset')

'''
Graph objects are dictionaries containing the following keys.
### Homogeneous graph:
- `edge_index` (necessary): `numpy.ndarray` of shape `(2, num_edges)`. Please include bidirectional edges explicitly if graphs are undirected.
- `num_nodes` (necessary): `int`, denoting the number of nodes in the graph.
- `node_feat` (optional): `numpy.ndarray` of shape `(num_nodes, node_feat_dim)`.
- `edge_feat` (optional): `numpy.ndarray` of shape `(num_edges, edge_feat_dim)`. 
'''

graph_list = []

graph_list.append(graph)

# saving a list of graphs
saver.save_graph_list(graph_list)

'''
## 4. Saving dataset split
Prepare `split_idx`, a dictionary with three keys, `train`, `valid`, and `test`, and mapping into data indices of `numpy.ndarray`. Then, call `saver.save_split(split_idx, split_name = xxx)`.
We need to split up the edges.
'''
split_idx = dict()
train_frac = args.subsample
if args.test_upto>0:
    if args.test_upto>num_edges:
        raise ValueError('test_upto larger than', num_edges)
    # split the first group (structured) edges into 20% test/valid and 80% train
    perm = np.random.permutation(args.test_upto)
    test_frac = 1.0-train_frac
    split_idx['test'] = perm[:int(test_frac*args.test_upto)]
    split_idx['valid'] = split_idx['test']
    split_idx['train'] = np.random.permutation(np.concatenate((perm[int(test_frac*args.test_upto):],np.arange(args.test_upto,num_edges))))
else:
    perm = np.random.permutation(num_edges)
    if train_frac>0.8:
        raise ValueError('train_frac>0.8 not yet implemented')
    valid_frac = train_frac+0.1
    test_frac = train_frac+0.2
    split_idx['train'] = perm[:int(train_frac*num_edges)]
    split_idx['valid'] = perm[int(train_frac*num_edges):int(valid_frac*num_edges)]
    split_idx['test'] = perm[int(valid_frac*num_edges):int(test_frac*num_edges)]

# need to generate the triples with these indices, graph is not otherwise used (?)
split_triples = { k: make_triples(graph,split_idx[k]) for k in split_idx.keys() }
# should add test negatives but do random ones now.
saver.save_split(split_triples, split_name = 'random')

'''
## 5. Copying mapping directory
Store all the mapping information and `README.md` in `mapping_path` and call `saver.copy_mapping_dir(mapping_path)`.

'''
mapping_path = 'mapping' + str(os.getpid()) + '/'

# prepare mapping information first and store it under this directory (empty below).
os.makedirs(mapping_path)
os.mknod(os.path.join(mapping_path, 'README.md'))
with open(os.path.join(mapping_path, 'args.txt'), mode='w') as out:
    print( args, file=out )
with open(os.path.join(mapping_path, 'split_idx.txt'), mode='w') as out:
    np.set_printoptions(threshold=sys.maxsize)
    for k in split_idx.keys():
        print( k, split_idx[k].tolist(), file=out )
    np.set_printoptions(threshold=100)
    
#os.system( 'cp ' + __file__ + ' generate_' + dataset_name + '.py' )
saver.copy_mapping_dir(mapping_path)

'''
## 6. Saving task information
Save task information by calling `saver.save_task_info(task_type, eval_metric, num_classes = num_classes)`.
`eval_metric` is used to call `Evaluator` (c.f. [here](https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py)). You can reuse one of the existing metrics, or you can implement your own by creating a pull request.
'''
saver.save_task_info(task_type = 'link prediction', eval_metric = 'mrr')
 
## 7. Getting meta information dictionary
meta_dict = saver.get_meta_dict()

## 8. Testing the dataset object
from ogb.linkproppred import LinkPropPredDataset
dataset = LinkPropPredDataset(dataset_name, meta_dict = meta_dict)

# see if it is working properly
print(dataset[0])
print(dataset.get_edge_split())

## 9. Zipping and cleaning up
#saver.zip()
#saver.cleanup()
