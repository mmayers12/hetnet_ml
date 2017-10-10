import re
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import diags
from hetio.hetnet import MetaGraph
from parallel import parallel_process
import graph_tools as gt
import matrix_tools as mt



class MatrixFormattedGraph(object):
    """
    Class for adjacency matrix representation of the heterogeneous network.
    """

    def __init__(self, nodes, edges, start_kind='Compound', end_kind='Disease',
                 max_length=4, metapaths_file=None, w=0.4):
        """
        Initializes the adjacency matrices used for feature extraction.

        :param nodes: DataFrame or string, location of the .csv file containing nodes formatted for neo4j import.
            This format must include two required columns: One column labeled ':ID' with the unique id for each 
            node, and one column named ':LABEL' containing the metanode type for each node
        :param edges: DataFrame or string, location of the .csv file containing edges formatted for neo4j import.
            This format must include three required columns: One column labeled  ':START_ID' with the node id
            for the start of the edge, one labeled ':END_ID' with teh node id for the end of the edge and one
            labeled ':TYPE' describing the metaedge type.
        :param start_kind: string, the source metanode. The node type from which the target edge to be predicted 
            as well as all metapaths originate.
        :param end_kind: string, the target metanode. The node type to which the target edge to be predicted 
            as well as all metapaths terminate.
        :param max_length: int, the maximum length of metapaths to be extracted by this feature extractor.
        :param metapaths_file: string, location of the metapaths.json file that contains information on all the
            metapaths to be extracted.  If provided, this will be used to generate metapath information and
            the variables `start_kind`, `end_kind` and `max_length` will be ignored.  
            This file must contain the following keys: 'edge_abbreviations' and 'standard_edge_abbreviations' which 
            matches the same format as ':TYPE' in the edges, 'edges' lists of each edge in the metapath. 
        :param w: float between 0 and 1. Dampening factor for producing degree-weighted matrices
        """
        # Store the values of the different files
        self.node_file = None
        self.edge_file = None

        if type(nodes) == str:
            self.node_file = nodes
        elif type(nodes) == pd.DataFrame:
            self.node_df = nodes.copy()
        if type(edges) == str:
            self.edge_file = edges
        elif type(edges) == pd.DataFrame:
            self.edge_df = edges.copy()

        self.metapaths_file = metapaths_file
        self.w = w
        self.metagraph = None

        # Read the information in the files
        print('Reading file information...')
        self.read_node_file()
        self.read_edge_file()
        if self.metapaths_file:
            self.read_metapaths_file()
        else:
            self.get_metapaths(start_kind, end_kind, max_length)

        # Generate the adjacency matrices.
        print('Generating adjacency matrices...')
        time.sleep(0.5)
        self.adj_matrices = self.generate_adjacency_matrices(self.metaedges)
        print('\nWeighting matrices by degree with dampening factor {}...'.format(w))
        time.sleep(0.5)
        self.degree_weighted_matrices = self.generate_weighted_matrices()

    def read_node_file(self):
        """Reads the nodes file and stores as a DataFrame, also generates a mapping dictionaries."""
        if self.node_file:
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
        """Reads the edge file and stores it as a DataFrame"""
        if self.edge_file:
            self.edge_df = pd.read_csv(self.edge_file, dtype={':START_ID': str, ':END_ID': str})
        self.edge_df = self.edge_df.dropna(subset=[':START_ID', ':END_ID', ':TYPE'])

        # Split the metaedge name from its abbreviation if both are included
        if all(self.edge_df[':TYPE'].str.contains('_')):
            e_types = self.edge_df[':TYPE'].unique()
            e_types_split = [e.split('_') for e in e_types]
            self.metaedges = [e[-1] for e in e_types_split]

            edge_abbrev_dict = {e: abv for e, abv in zip(e_types, self.metaedges)}
            self.edge_df['abbrev'] = self.edge_df[':TYPE'].apply(lambda t: edge_abbrev_dict[t])

        ##%%TODO Ideally all eges should have their abbreviations included, so this should never be run...
        else:
            self.metaedges = self.edge_df[':TYPE'].unique()
            self.edge_df['abbrev'] = self.edge_df[':TYPE']

    def read_metapaths_file(self):
        """Reads metapaths.json file if one is given and stores."""
        # Read the metapaths
        with open(self.metapaths_file) as fin:
            mps = json.load(fin)

        # Reformat the metapaths to a dict so that the abbreviation is the key.
        self.metapaths = dict()
        for mp in mps:
            self.metapaths[mp['abbreviation']] = {k:v for k, v in mp.items() if k != 'abbreviation'}

    def get_metagraph(self):
        """Generates class variable metagraph, an instance of hetio.hetnet.MetaGraph"""

        print('Initializing metagraph...')
        time.sleep(0.5)
        abbrev_dict, edge_tuples = gt.get_abbrev_dict_and_edge_tuples(self.node_df, self.edge_df)
        self.metagraph = MetaGraph.from_edge_tuples(edge_tuples, abbrev_dict)

    def get_metapaths(self, start_kind, end_kind, max_length):
        """Generates the class variable metapaths, which has information on each of the meatpaths to be extracted"""
        if not self.metagraph:
            self.get_metagraph()

        metapaths = self.metagraph.extract_metapaths(start_kind, end_kind, max_length) 

        self.metapaths = dict()
        for mp in metapaths:
            if len(mp) == 1:
                continue
            mp_info = dict()
            mp_info['length'] = len(mp)
            mp_info['edges'] = [str(x) for x in mp.edges]
            mp_info['edge_abbreviations'] = [x.get_abbrev() for x in mp.edges]
            mp_info['standard_edge_abbreviations'] = [x.get_standard_abbrev() for x in mp.edges]
            self.metapaths[str(mp)] = mp_info

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
        start = self.edge_df[self.edge_df['abbrev'] == metaedge][':START_ID'].apply(lambda s: self.nid_to_index[s])
        end = self.edge_df[self.edge_df['abbrev'] == metaedge][':END_ID'].apply(lambda s: self.nid_to_index[s])

        # Add edges to the matrix
        for s, e in zip(start, end):
            mat[s, e] = 1
            if not directed:
                mat[e, s] = 1

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

    def generate_weighted_matrices(self):
        """Generates the weighted matrices for DWPC and DWWC calculation"""
        weighted_matrices = dict()
        for metaedge, matrix in tqdm(self.adj_matrices.items()):
            # Directed
            if '>' in metaedge or '<' in metaedge:
                weighted_matrices[metaedge] = mt.weight_by_degree(matrix, w=self.w, directed=True)
            # Undirected
            else:
                weighted_matrices[metaedge] = mt.weight_by_degree(matrix, w=self.w, directed=False)
        return weighted_matrices

    def validate_ids(self, ids):
        """
        Ensures that a given id is either a Node type, list of node ids, or list of node indices.

        :param ids: string or list of strings or ints. The ids to be validated
        :return: list, the indicies corresponding to the ids in the matrices.
        """
        if type(ids) == str:
            return self.metanode_idxs[ids]
        elif type(ids) == list:
            if type(ids[0]) == str:
                return [self.nid_to_index[nid] for nid in ids]
            elif type(ids[0]) == int:
                return ids

        raise ValueError()

    def prep_node_info_for_extraction(self, start_nodes=None, end_nodes=None):
        """
        Given a start and end node type or list of ids or indices finds the following:
            The indices for the nodes
            The metanode types for the nodes
            The node id's for the nodes
            A column name for the node type (type.lower() + '_id')

        :param start_nodes: string or list, title of the metanode of the nodes for the start of the metapaths.
            If a list, can be IDs or indicies corresponding to a subset of starting nodes for the feature.
        :param end_nodes: String or list, String title of the metanode of the nodes for the end of the metapaths.
            If a list, can be IDs or indicies corresponding to a subset of ending nodes for the feature.

        :returns: start_idxs, end_idxs, start_type, end_type, start_ids, end_ids, start_name, end_name
            lists and strings with information on the starting and ending nodes.
        """

        # Validate that we either have a list of nodeids, a list of indices, or a string of metanode
        start_idxs = self.validate_ids(start_nodes)
        end_idxs = self.validate_ids(end_nodes)

        # Get metanode types for start and end
        start_type = self.idx_to_metanode[start_idxs[0]]
        end_type = self.idx_to_metanode[end_idxs[0]]

        # Get the ids and names for the start and end to initialize DataFrame
        start_ids = [self.index_to_nid[x] for x in start_idxs]
        end_ids = [self.index_to_nid[x] for x in end_idxs]
        start_name = start_type.lower() + '_id'
        end_name = end_type.lower() + '_id'

        return start_idxs, end_idxs, start_type, end_type, start_ids, end_ids, start_name, end_name

    def process_extraction_results(self, result, metapaths, start_nodes, end_nodes):
        """
        Given a list of matrices and a list of metapaths, processes the feature extraction into a pandas DataFrame.


        :param result: list of matrices, the feature extraction result.
        :param metapaths: list of strings, the names of the metapaths extracted
        :param start_nodes: string or list, title of the metanode of the nodes for the start of the metapaths.
            If a list, can be IDs or indicies corresponding to a subset of starting nodes for the feature.
        :param end_nodes: string or list, String title of the metanode of the nodes for the start of the metapaths.
            If a list, can be IDs or indicies corresponding to a subset of starting nodes for the feature.

        :return: pandas.DataFrame, with columns for start_node_id, end_node_id and columns for each of the metapath 
            features calculated.
        """
        from itertools import product

        # Get information on nodes needed for DataFrame formatting.
        start_idxs, end_idxs, start_type, end_type, start_ids, end_ids, start_name, end_name = \
            self.prep_node_info_for_extraction(start_nodes, end_nodes)

        # Slice the resultant matrices to find only start and end results
        print('\nSub-setting resultant matrices...')
        time.sleep(0.5)

        result = [r[start_idxs, :][:, end_idxs] for r in tqdm(result)]

        # Turn each result matrix into a series
        print('\nFormatting results to series...')
        time.sleep(0.5)
        results = []

        # Currently running in series.  Extensive testing has found no incense in speed via Parallel processing
        # However, parallel usually results in an inaccurate counter.
        for i in tqdm(range(len(metapaths))):
            results.append(mt.to_series(result[i], name=metapaths[i]).reset_index(drop=True))

        # Past all the series together into a DataFrame
        print('\nConcatenating series to DataFrame...')
        start_end_df = pd.DataFrame(list(product(start_ids, end_ids)), columns=[start_name, end_name])

        return pd.concat([start_end_df]+results, axis=1)

    def extract_dwpc(self, metapaths=None, start_nodes=None, end_nodes=None, verbose=False, n_jobs=1):
        """
        Extracts DWPC metrics for the given metapaths.  If no metapaths are given, will calcualte for all metapaths.

        :param metapaths: list or None, the metapaths paths to calculate DWPC values for.  List must be a subset of
            those found in metapahts.json.  If None, will calcualte DWPC values for metapaths in the metapaths.json
            file.
        :param start_nodes: String or list, String title of the metanode start of the metapaths.
            If a list, can be IDs or indicies corresponding to a subset of starting nodes for the DWPC.
        :param end_nodes: String or list, String title of the metanode for the end of the metapaths.  If a
            list, can be IDs or indicies corresponding to a subset of ending nodes for the DWPC.
        :param verbose: boolean, if True, prints debugging text for calculating each DWPC. (not optimized for 
            parallel processing).
        :param n_jobs: int, the number of jobs to use for parallel processing.

        :return: pandas.DataFrame, Table of results with columns corresponding to DWPC values from start_id to 
            end_id for each metapath.
        """

        # If not given a list of metapaths, calculate for all
        if not metapaths:
            metapaths = sorted(list(self.metapaths.keys()))

        # Validate the ids before running the calculation
        self.validate_ids(start_nodes)
        self.validate_ids(end_nodes)

        print('Calculating DWPCs...')
        time.sleep(0.5)

        # Prepare functions for parallel processing
        arguments = []
        for mp in metapaths:
            path = mt.get_path(mp, self.metapaths)
            to_multiply = mt.get_matrices_to_multiply(path, self.degree_weighted_matrices)
            edges = mt.get_edge_names(mp, self.metapaths)
            arguments.append({'path': path, 'edges': edges, 'to_multiply': to_multiply, 'verbose': verbose})

        # Run DPWC calculation processes in parallel
        result = parallel_process(array=arguments, function=mt.count_paths, use_kwargs=True, 
                                  n_jobs=n_jobs, front_num=0)
        del(arguments)

        # Process and return results
        results = self.process_extraction_results(result, metapaths, start_nodes, end_nodes)
        return results

    def extract_dwwc(self, metapaths=None, start_nodes=None, end_nodes=None, n_jobs=1):
        """
        Extracts DWWC metrics for the given metapaths.  If no metapaths are given, will calcualte for all metapaths.

        :param metapaths: list or None, the metapaths paths to calculate DWPC values for.  List must be a subset of
            those found in metapahts.json.  If None, will calcualte DWPC values for all metapaths in the 
            metapaths.json file.
        :param start_nodes: String or list, String title of the metanode for the start of the metapaths.
            If a list, can be IDs or indicies corresponding to a subset of starting nodes for the DWWC.
        :param end_nodes: String or list, String title of the metanode for the end of the metapaths.  If a
            list, can be IDs or indicies corresponding to a subset of ending nodes for the DWWC.
        :param n_jobs: int, the number of jobs to use for parallel processing.

        :return: pandas.DataFrame. Table of results with columns corresponding to DWPC values from start_id to 
            end_id for each metapath.
        """

        # If not given a list of metapaths, calculate for all
        if not metapaths:
            metapaths = list(self.metapaths.keys())

        # Validate the ids before running the calculation
        self.validate_ids(start_nodes)
        self.validate_ids(end_nodes)

        print('Calculating DWWCs...')
        time.sleep(0.5)

        # Prepare functions for parallel processing
        arguments = []
        for mp in metapaths:
            arguments.append({'path': mt.get_path(mp, self.metapaths), 'matrices': self.degree_weighted_matrices})
        for mp in metapaths:
            path = mt.get_path(mp, self.metapaths)
            to_multiply = mt.get_matrices_to_multiply(path, self.degree_weighted_matrices)
            arguments.append({'path': path, 'to_multiply': to_multiply})
        # Run DWWC calculation processes in parallel
        result = parallel_process(array=arguments, function=mt.count_walks, use_kwargs=True, 
                                  n_jobs=n_jobs, front_num=0)

        # process and resturn results
        results = self.process_extraction_results(result, metapaths, start_nodes, end_nodes)
        return results


    def extract_degrees(self, start_nodes=None, end_nodes=None, subset=None):
        """
        Extracts degree features from the metagraph

        :param start_nodes: string or list, title of the metanode (string) from which paths originate, or a list of
            either IDs or indicies corresponding to a subset of starting nodes.
        :param end_nodes: string or list, title of the metanode (string) at which paths terminate, or a list of
            either IDs or indicies corresponding to a subset of ending nodes.
        :return:
        """
        # Get all node info needed to process results.
        start_idxs, end_idxs, start_type, end_type, start_ids, end_ids, start_name, end_name = \
            self.prep_node_info_for_extraction(start_nodes, end_nodes)

        # Initialize multi-index dataframe for results
        index = pd.MultiIndex.from_product([start_ids, end_ids], names=[start_name, end_name])
        result = pd.DataFrame(index=index)

        def get_edge_abbrev(kind, edge):
            if kind == edge.get_id()[0]:
                return edge.get_abbrev()
            elif kind == edge.get_id()[1]:
                return edge.inverse.get_abbrev()
            else:
                return None

        # Make the metagraph if not already initialized
        if not self.metagraph:
            self.get_metagraph()

        # Loop through each edge in the metagraph
        if not subset:
            edges = list(self.metagraph.get_edges())
        else:
            edges = [e for e in self.metagraph.get_edges() if e.get_abbrev() in subset]
        for edge in tqdm(edges):

                # Get those edges that link to start and end metanodes
                start_abbrev = get_edge_abbrev(start_type, edge)
                end_abbrev = get_edge_abbrev(end_type, edge)
                abbrev = edge.get_abbrev()

                # Calculate the degrees
                degrees = (self.adj_matrices[abbrev] * self.adj_matrices[abbrev]).diagonal()

                # Add degree results to DataFrame
                if start_abbrev:
                    start_series = pd.Series(degrees[start_idxs], index=start_ids, dtype='int64')
                    result = result.reset_index()
                    result = result.set_index(start_name)
                    result[start_abbrev] = start_series

                if end_abbrev:
                    end_series = pd.Series(degrees[end_idxs], index=end_ids, dtype='int64')
                    result = result.reset_index()
                    result = result.set_index(end_name)
                    result[end_abbrev] = end_series

        # Sort Columns by Alpha
        result = result.reset_index()
        cols = sorted([c for c in result.columns if c not in [start_name, end_name]])
        return result[[start_name, end_name]+cols]

    def extract_prior_estimate(self, edge, start_nodes=None, end_nodes=None):
        """
        Estimates the prior probability that a target edge exists between a given source and target node pair.
        Prior probability is dependant on the degrees of the nodes across the given edge.
        Further discussion here: https://think-lab.github.io/d/201/
        
        :param edge: string, the abbreviation of the metaedge across which the prior probability will be extracted.
            e.g. 'CtD' for the Compound-treats-Disease edge.
        :return: pandas.DataFrame, with all combinations of instances of the Start and End metanodes for the edge,
            with a prior probability of the edge linking them.
        """

        def estimate_prior(out_degree, in_degree, total):
            """
            Attempts to calculate the prior probability via the following formula:

            for example if there are 755 Compund-treats-Disease edges
            compound A has 3 compound-treats-disease edges
            disease Z has 5 compound-treats-diease edges

            Prob that compound A treats disease Z = 
                1 - (750/755 * 749/754 * 748/753)
            """

            res = 1
            for i in range(out_degree):
                res *= ((total - i - in_degree) / (total - i))

            return 1 - res

        # Ensure the metagraph has be initialized
        if not self.metagraph:
            self.metagraph = self.get_metagraph()

        # Extract the source and target information from the metagraph
        mg_edge = self.metagraph.metapath_from_abbrev(edge).edges[0]
        if not start_nodes:
            start_nodes = mg_edge.source.get_id()
        if not end_nodes:
            end_nodes = mg_edge.target.get_id()

        # Get degrees across edge, which are needed for this computation
        print('\nExtracting degrees along edge {}'.format(edge))
        time.sleep(0.5)
        degrees = self.extract_degrees(start_nodes, end_nodes, edge)

        # Compute the prior for each pair
        print('\nComputing Prior...')
        time.sleep(0.5)
        subset = degrees.iloc[:, 0].name
        total = degrees.drop_duplicates(subset=subset)[edge].sum()
        result = []
        for row in tqdm(degrees.itertuples(index=False), total=len(degrees)):
            result.append(estimate_prior(row[2], row[3], total))
        degrees['prior'] = result

        # the sum of the probabilities must add up to the total number of positives
        weighting_factor = total / degrees['prior'].sum()
        degrees['prior'] = degrees['prior'] * weighting_factor

        return_cols = [c for c in degrees.columns if c.endswith('_id')] + ['prior']

        return degrees[return_cols]


