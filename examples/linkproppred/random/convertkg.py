from ogb.io import DatasetSaver
import numpy as np
import networkx as nx
import argparse
import os
import csv

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Convert Knowledge Graph Formats',
        usage='convertkg.py [<args>] [-h | --help]'
    )

    parser.add_argument('dataset', type=str)
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-m', '--mode', type=str)
    parser.add_argument('-ep', '--edge_probability', type=float, default=0.1)
    parser.add_argument('-nv', '--n_vertices', type=int)
    parser.add_argument('-nr', '--n_relations', type=int)
    parser.add_argument('--map_node_file', type=str)
    parser.add_argument('--map_relation_file', type=str)
    return parser.parse_args(args)

def read_map(file):
    if file=='':
        return None
    map = dict()
    with open(file, newline='') as csvfile:
        csvfile.readline()
        for (idx,name) in csv.reader(csvfile, delimiter=',', quotechar='|'):
            map[name] = int(idx)
    return map

args = parse_args()
dataset_name = args.dataset
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
    print(graph)
else:
    raise ValueError('unknown mode')

#### should not need to modify below this line

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
num_edges = graph['edge_index'].shape[1]
perm = np.random.permutation(num_edges)
split_idx['train'] = perm[:int(0.8*num_edges)]
split_idx['valid'] = perm[int(0.8*num_edges): int(0.9*num_edges)]
split_idx['test'] = perm[int(0.9*num_edges):]
# need to generate the triples with these indices, graph is not otherwise used (?)
split_triples = { k: make_triples(graph,split_idx[k]) for k in split_idx.keys() }
# should add test negatives but do random ones now.
saver.save_split(split_triples, split_name = 'random')

'''
## 5. Copying mapping directory
Store all the mapping information and `README.md` in `mapping_path` and call `saver.copy_mapping_dir(mapping_path)`.

'''
mapping_path = 'mapping/'

# prepare mapping information first and store it under this directory (empty below).
os.makedirs(mapping_path)
os.mknod(os.path.join(mapping_path, 'README.md'))
os.system( 'cp ' + __file__ + ' generate_' + dataset_name + '.py' )
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
