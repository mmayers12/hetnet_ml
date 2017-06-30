import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import diags
import matrix_tools as mt


class MatrixFormattedGraph(object):
    """
    Class for adjacency matrix representation of the heterogeneious network. 
    """

    def __init__(self, node_file, edge_file, metapaths_file, w=0.4):
        """
        Class for adjacency matrix representation of the heterogeneious network.
        
        :param node_file: string, location of the .csv file containing nodes formatted for neo4j import.  
            This format must include two required columns: One column labeled ':ID' with the unique id for each node, 
            and one column named ':LABEL' containing the metanode type for each node
        :param edge_file: string, location of the .csv file containing edges formatted for neo4j import.
            This format must inculde three required columns: One column labeled  ':START_ID' with the node id
            for the start of the edge, one labeled ':END_ID' with teh node id for the end of the edge and one
            labeled ':TYPE' descrbing the metaedge type.
        :param metapaths_file: string, location of the metapaths.json file that contains information on all the
            metapaths to be extracted.  This file must contain the following keys: 'edge_abbreviation' which matches 
            the same format as ':TYPE' in the edge_file, 'edges' lists of each edge in the metapath
        :param w: float between 0 and 1. Dampening factor for producing degree-weighted matricies
        """
        # Store the values of the different files
        self.node_file = node_file
        self.edge_file = edge_file
        self.metapaths_file = metapaths_file
        self.w = w

        # Read the information in the files
        print('Reading file information...')
        self.read_node_file()
        self.read_edge_file()
        self.read_metapaths_file()

        # Generate the adjacency matrices.
        print('Generating adjcency matrices...')
        self.adj_matrices = self.generate_adjacency_matrices(self.metaedges)
        print('\nWeighting matrices by degree with dampening factor {}...'.format(w))
        self.degree_weighted_matrices = self.generate_weighted_matrices(self.adj_matrices, self.w)

    def read_node_file(self):
        self.node_df = pd.read_csv(self.node_file, dtype={':ID': str})
        self.nodes = self.node_df[':ID']

        # Get mapping from id to index and reverse
        self.index_to_nid = self.nodes.to_dict()
        self.nid_to_index = pd.Series(self.nodes.index.values, index=self.nodes).to_dict()

        # Get mapping from metanodes to a list of indices
        self.metanode_idxs = dict()
        for kind in self.node_df[':LABEL'].unique():
            self.metanode_idxs[kind] = list(self.node_df[self.node_df[':LABEL'] == kind].index)
        self.idx_to_metanode = self.node_df[':LABEL'].to_dict()

    def read_edge_file(self):
        self.edge_df = pd.read_csv(self.edge_file, dtype={':START_ID': str, ':END_ID': str})

        # Fix the edge abbreviations if full name is included
        if sum(self.edge_df[':TYPE'].str.contains('_')) != 0:
            self.edge_df[':TYPE'] = self.edge_df[':TYPE'].str.split('_', expand=True).iloc[:, -1]

        self.metaedges = self.edge_df[':TYPE'].unique()

    def read_metapaths_file(self):
        # Read the metapaths
        with open(self.metapaths_file) as fin:
            mps = json.load(fin)

        # Reformat the metapaths to a dict so that the abbreviation is the key.
        self.metapaths = dict()
        for mp in mps:
            self.metapaths[mp['abbreviation']] = {k:v for k, v in mp.items() if k != 'abbreviation'}

    def get_adj_matrix(self, metaedge, directed=False):
        """
        Create a sparse adjacency matrix for the given metaedge.

        :param metaedge: String, the abbreviation for the metadge. e.g. 'CbGaD'
        :param directed: Boolean, If True, only adds edges in the forward direction. (Default: False)

        :return: Sparse matrix, adjacency matrix for the given metaedge. Row and column indices correspond
            to the order of the node ids in the global variable `nodes`.
        """

        # Fast generation of a sparse matrix of zeroes
        mat = np.zeros((1, len(self.nodes)), dtype='int16')[0]
        mat = diags(mat).tolil()

        # Find the start and end nodes for edges of the given type
        start = self.edge_df[self.edge_df[':TYPE'] == metaedge][':START_ID'].apply(lambda s: self.nid_to_index[s])
        end = self.edge_df[self.edge_df[':TYPE'] == metaedge][':END_ID'].apply(lambda s: self.nid_to_index[s])

        # Add edges to the matrix
        for s, e in zip(start, end):
            mat[s, e] += 1
            if not directed:
                mat[e, s] += 1

        return mat.tocsc()

    def generate_adjacency_matrices(self, metaedges):
        """
        Generates adjacency matrices for performing path and walk count operations.

        :param metaedges: list, the names of edge types to be used
        :returns: adjacency_matrices, dictionary of sparse matrices with keys corresponding to the metaedge type.
        """
        adjacency_matrices = dict()

        for metaedge in tqdm(metaedges):
            # Directed in forward direction
            if '>' in metaedge:
                adjacency_matrices[metaedge] = self.get_adj_matrix(metaedge, directed=True)
                reverse = mt.get_reverse_directed_edge(metaedge)
                adjacency_matrices[reverse] = self.get_adj_matrix(metaedge, directed=True).transpose()
            # Undirected
            else:
                adjacency_matrices[metaedge] = self.get_adj_matrix(metaedge, directed=False)

        return adjacency_matrices

    def generate_weighted_matrices(self, adj_matrices, w):

        weighted_matrices = dict()
        for metaedge, matrix in tqdm(adj_matrices.items()):
            # Directed
            if '>' in metaedge or '<' in metaedge:
                weighted_matrices[metaedge] = mt.weight_by_degree(matrix, w=w, directed=True)
            # Undirected
            else:
                weighted_matrices[metaedge] = mt.weight_by_degree(matrix, w=w, directed=False)
        return weighted_matrices

    def validate_ids(self, ids):
        """
        Ensures that a given id is either a Node type, list of node ids, or list of node indices.
        
        :param ids: string or list of strings or ints. The ids to be validated 
        :return: 
        """
        if type(ids) == str:
            return self.metanode_idxs[ids]
        elif type(ids) == list:
            if type(ids[0]) == str:
                return [self.nid_to_index[nid] for nid in ids]
            elif type(ids[0]) == int:
                return ids

        raise ValueError()

    def calculate_dwpc(self, metapaths=None, start_nodes=None, end_nodes=None, verbose=False, n_jobs=1):
        """
        
        :param metapaths: 
        :param start_nodes: 
        :param end_nodes: 
        :param verbose: 
        :return: 
        """
        from parallel import parallel_process

        # If not given a list of metapaths, calculate for all
        if not metapaths:
            metapaths = list(self.metapaths.keys())

        # Validate that we either have a list of nodeids, a list of indices, or a string of metanode
        start_nodes = self.validate_ids(start_nodes)
        end_nodes = self.validate_ids(end_nodes)

        start_type = self.idx_to_metanode[start_nodes[0]]
        end_type = self.idx_to_metanode[end_nodes[0]]

        print('Calculating DWPCs...')

        # Prepare functions for parallel processing
        arguments = []
        mps = list(self.metapaths.keys())
        for mp in mps:
            arguments.append({'metapath':mp, 'metapaths': self.metapaths, 'verbose': verbose,
                              'matrices': self.degree_weighted_matrices})
        # Run DPWC calculation processes in parallel
        result = parallel_process(array=arguments, function=mt.count_paths, use_kwargs=True, n_jobs=n_jobs, front_num=0)

        # Format the matrices into a DataFrame
        print('\nReformating results...')
        # Turn to a dictionary
        dwpcs = {mp: res for mp, res in zip(mps, result)}
        results = pd.DataFrame()
        for metapath, dwpc in tqdm(dwpcs.items()):
            results[metapath] = self.to_series(dwpc, start_nodes=start_nodes, end_nodes=end_nodes)

        # Fix column names to correspond to proper metanode_ids
        start_name = start_type.lower() + '_id'
        end_name = end_type.lower() + '_id'

        results = results.reset_index(drop=False)
        results = results.rename(columns={'level_0': start_name, 'level_1': end_name})

        return results

    def calculate_degrees(self, start_nodes=None, end_nodes=None):
        """
        
        :param metapaths: 
        :param start_nodes: 
        :param end_nodes: 
        :return: 
        """
        # Validate that we either have a list of nodeids, a list of indices, or a string of metanode
        start_nodes = self.validate_ids(start_nodes)
        end_nodes = self.validate_ids(end_nodes)

        start_type = self.idx_to_metanode[start_nodes[0]]
        end_type = self.idx_to_metanode[end_nodes[0]]

        result = pd.DataFrame()

        for edge in tqdm(self.metaedges):
            degrees = self.adj_matrices[edge] * self.adj_matrices[edge]
            if degrees[start_nodes, :][:, start_nodes].sum() > 0:
                degree_for = start_nodes
            elif degrees[start_nodes, :][:, start_nodes].sum() > 0:
                degree_for = end_nodes
            else:
                continue

            node_type = self.idx_to_metanode[degree_for[0]]
            if node_type[0] == edge[0]:
                result[edge] = self.to_series(degrees, node_type, node_type)
            else:
                edge_name = mt.get_reverse_undirected_edge(edge)
                result[edge_name] = self.to_series(degrees, node_type, node_type)

        start_name = start_type.lower() + '_id'
        end_name = end_type.lower() + '_id'

        result = result.reset_index(drop=False)
        result = result.rename(columns={'level_0': start_name, 'level_1': end_name})
        return result

    def to_series(self, dwpc, start_nodes, end_nodes):
        """
        
        :param dwpc: 
        :param start_nodes: 
        :param end_nodes: 
        :return: 
        """
        dat = pd.DataFrame(dwpc.todense()[start_nodes, :][:, end_nodes],
                           index=[self.index_to_nid[sid] for sid in start_nodes],
                           columns=[self.index_to_nid[eid] for eid in end_nodes])
        return dat.stack()
