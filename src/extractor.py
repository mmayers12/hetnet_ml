import re
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import diags
from hetio.hetnet import MetaGraph
from parallel import parallel_process
import matrix_tools as mt


class MatrixFormattedGraph(object):
    """
    Class for adjacency matrix representation of the heterogeneous network.
    """

    def __init__(self, node_file, edge_file, start_kind='Compound', end_kind='Disease',
                 max_length=4, metapaths_file=None, w=0.4):
        """
        Initializes the adjacency matrices used for feature extraction.

        :param node_file: string, location of the .csv file containing nodes formatted for neo4j import.
            This format must include two required columns: One column labeled ':ID' with the unique id for each node,
            and one column named ':LABEL' containing the metanode type for each node
        :param edge_file: string, location of the .csv file containing edges formatted for neo4j import.
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
            matches the same format as ':TYPE' in the edge_file, 'edges' lists of each edge in the metapath. 
        :param w: float between 0 and 1. Dampening factor for producing degree-weighted matrices
        """
        # Store the values of the different files
        self.node_file = node_file
        self.edge_file = edge_file
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
        self.edge_df = pd.read_csv(self.edge_file, dtype={':START_ID': str, ':END_ID': str})

        # Split the metaedge name from its abbreviation if both are included
        if sum(self.edge_df[':TYPE'].str.contains('_')) != 0:
            long_abbrev = self.edge_df[':TYPE'].str.split('_', expand=True)
            self.metaedge_names = long_abbrev.iloc[:, 0].unique()
            self.metaedges = long_abbrev.iloc[:, 1].unique()
            self.edge_df['abbrev'] = long_abbrev.iloc[:, 1]

        else:
            self.metaedge_names = None
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

        def get_tuples(start_ids, end_ids, types):
            """Generates metaedge tuples form node start_ids, end_ids and types"""
            def get_direction(t):
                """Finds the direction of a metaedge from its abbreviaton"""
                if '>' in t:
                    return 'forward'
                elif '<' in t:
                    return 'backward'
                else:
                    return 'both'

            start_kinds = start_ids.apply(lambda s: self.idx_to_metanode[self.nid_to_index[s]])
            end_kinds = end_ids.apply(lambda e: self.idx_to_metanode[self.nid_to_index[e]])

            tuples_df = pd.DataFrame()
            tuples_df['start_kind'] = start_kinds
            tuples_df['end_kind'] = end_kinds
            tuples_df['types'] = types

            tuples_df = tuples_df.drop_duplicates()

            tuples_df['edge'] = tuples_df['types'].str.split('_', expand=True)[0]
            tuples_df['direction'] = tuples_df['types'].apply(get_direction)

            tuple_columns = ['start_kind', 'end_kind', 'edge', 'direction']

            tuples = [tuple(r) for r in tuples_df[tuple_columns].itertuples(index=False)]

            return tuples

        def get_abbrev_dict():
            """Returns an abbreviation dictionary generated from class variables."""
            node_kinds = self.node_df[':LABEL'].unique()
            edge_kinds = self.metaedge_names

            node_abbrevs = [''.join([w[0].upper() for w in re.split('[ -]', t)]) for t in node_kinds]
            edge_abbrevs = [''.join([w[0].lower() for w in t.split('-')]) for t in edge_kinds]

            node_abbrev_dict = {t: a for t, a in zip(node_kinds, node_abbrevs)}
            edge_abbrev_dict = {t: a for t, a in zip(edge_kinds, edge_abbrevs)}

            # Fix an edge case wrt Rephetio
            if 'Pathway' in node_abbrev_dict:
                node_abbrev_dict['Pathway'] = 'PW'

            return {**node_abbrev_dict, **edge_abbrev_dict}

        print('Initializing metagraph...')
        edge_tuples = get_tuples(self.edge_df[':START_ID'], self.edge_df[':END_ID'], self.edge_df[':TYPE'])
        abbrev_dict = get_abbrev_dict()

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
        :return: list, the indicies corresponding to the ids in the matricies.
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
        
        :param start_nodes: string or list, String title of the metanode of the nodes for the start of the metapaths.
            If a list, can be IDs or indicies corresponding to a subset of starting nodes for the feature.
        :param end_nodes: String or list, String title of the metanode of the nodes for the end of the metapaths.  If a
            list, can be IDs or indicies corresponding to a subset of ending nodes for the feature.
        
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
        Given a list of matricies and a list of metapaths, processes the feature extraction into a pandas DataFrame.
    
        
        :param result: list of matricies, the feature extraction result. 
        :param metapaths: list of strings, the names of the metapaths extracted
        :param start_nodes: string or list, String title of the metanode of the nodes for the start of the metapaths.
            If a list, can be IDs or indicies corresponding to a subset of starting nodes for the feature.
        :param end_nodes: string or list, String title of the metanode of the nodes for the start of the metapaths.
            If a list, can be IDs or indicies corresponding to a subset of starting nodes for the feature.
        
        :return: pandas.DataFrame, with columns for start_node_id, end_node_id, and columns for each of the metapath 
            features calculated. 
        """
        # Get information on nodes needed for DataFrame formatting.
        start_idxs, end_idxs, start_type, end_type, start_ids, end_ids, start_name, end_name = \
            self.prep_node_info_for_extraction(start_nodes, end_nodes)

        # Format the matrices into a DataFrame
        print('\nReformating results...')
        time.sleep(0.5)

        results = pd.DataFrame()

        # Currently running in series.  Extensive testing has found no incense in speed via Parallel processing
        # However, parallel usually results in an inaccurate counter.
        for mp, dwpc in tqdm(zip(metapaths, result), total=len(metapaths)):
            results[mp] = mt.to_series(dwpc, start_idxs, end_idxs, self.index_to_nid, name=mp)

        results = results.reset_index(drop=False)
        results = results.rename(columns={'level_0': start_name, 'level_1': end_name})

        return results

    def extract_dwpc(self, metapaths=None, start_nodes=None, end_nodes=None, verbose=False, n_jobs=1):
        """
        Extracts DWPC metrics for the given metapaths.  If no metapaths are given, will calcualte for all metapaths.

        :param metapaths: list or None, the metapaths paths to calculate DWPC values for.  List must be a subset of
            those found in metapahts.json.  If None, will calcualte DWPC values for all metapaths in the metapaths.json
            file.
        :param start_nodes: String or list, String title of the metanode of the nodes for the start of the metapaths.
            If a list, can be IDs or indicies corresponding to a subset of starting nodes for the DWPC.
        :param end_nodes: String or list, String title of the metanode of the nodes for the end of the metapaths.  If a
            list, can be IDs or indicies corresponding to a subset of ending nodes for the DWPC.
        :param verbose: boolean, if True, prints debugging text for calculating each DWPC. (not optimized for parallel
            processing).
        :param n_jobs: int, the number of jobs to use for parallel processing.

        :return: pandas.DataFrame. Table of results with columns corresponding to DWPC values from start_id to end_id
            for each metapath.
        """

        # If not given a list of metapaths, calculate for all
        if not metapaths:
            metapaths = list(self.metapaths.keys())

        # Validate the ids before running the calculation
        self.validate_ids(start_nodes)
        self.validate_ids(end_nodes)

        print('Calculating DWPCs...')
        time.sleep(0.5)

        # Prepare functions for parallel processing
        arguments = []
        for mp in metapaths:
            arguments.append({'path': mt.get_path(mp, self.metapaths), 'edges': mt.get_edge_names(mp, self.metapaths),
                              'verbose': verbose, 'matrices': self.degree_weighted_matrices})

        # Run DPWC calculation processes in parallel
        result = parallel_process(array=arguments, function=mt.count_paths, use_kwargs=True, n_jobs=n_jobs, front_num=0)

        # Process and return results
        results = self.process_extraction_results(result, metapaths, start_nodes, end_nodes)
        return results

    def extract_dwwc(self, metapaths=None, start_nodes=None, end_nodes=None, n_jobs=1):
        """
        Extracts DWWC metrics for the given metapaths.  If no metapaths are given, will calcualte for all metapaths.

        :param metapaths: list or None, the metapaths paths to calculate DWPC values for.  List must be a subset of
            those found in metapahts.json.  If None, will calcualte DWPC values for all metapaths in the metapaths.json
            file.
        :param start_nodes: String or list, String title of the metanode of the nodes for the start of the metapaths.
            If a list, can be IDs or indicies corresponding to a subset of starting nodes for the DWWC.
        :param end_nodes: String or list, String title of the metanode of the nodes for the end of the metapaths.  If a
            list, can be IDs or indicies corresponding to a subset of ending nodes for the DWWC.
        :param n_jobs: int, the number of jobs to use for parallel processing.

        :return: pandas.DataFrame. Table of results with columns corresponding to DWPC values from start_id to end_id
            for each metapath.
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
        # Run DWWC calculation processes in parallel
        result = parallel_process(array=arguments, function=mt.count_walks, use_kwargs=True, n_jobs=n_jobs, front_num=0)

        # process and resturn results
        results = self.process_extraction_results(result, metapaths, start_nodes, end_nodes)
        return results

    def extract_degrees(self, start_nodes=None, end_nodes=None):
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
        edges = list(self.metagraph.get_edges())
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
