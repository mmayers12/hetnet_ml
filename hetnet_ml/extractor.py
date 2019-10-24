import time
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import lil_matrix, hstack
from hetio.hetnet import MetaGraph
from .parallel import parallel_process
from . import graph_tools as gt
from . import matrix_tools as mt


class MatrixFormattedGraph(object):
    """
    Class for adjacency matrix representation of the heterogeneous network.
    """

    def __init__(self, nodes, edges, start_kind='Compound', end_kind='Disease', max_length=4, w=0.4, n_jobs=1):
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
        :param w: float between 0 and 1. Dampening factor for producing degree-weighted matrices
        :param n_jobs: int, the number of jobs to use for parallel processing.
        """
        # Initialize important class variables
        self.w = w
        self.n_jobs = n_jobs
        self.metagraph = None
        self.start_kind = start_kind
        self.end_kind = end_kind

        # Placeholders for variables to be defined later
        self.node_file = None
        self.edge_file = None
        self.nodes = None
        self.metaedges = None
        self.adj_matrices = None
        self.out_degree = dict()
        self.in_degree = dict()
        self.degree_weighted_matrices = None

        # Mappers to be used later
        self.nid_to_index = None
        self.index_to_nid = None
        self.id_to_metanode = None
        self.metanode_to_ids = None
        self.nid_to_name = None
        self.metanode_to_edges = dict()

        # Read and/or store nodes as DataFrame
        if type(nodes) == str:
            self.node_file = nodes
            print('Reading file information...')
            self._read_node_file()
        elif type(nodes) == pd.DataFrame:
            self.node_df = gt.remove_colons(nodes).copy()
        self._validate_nodes()

        # Read and/or store edges as DataFrame
        if type(edges) == str:
            self.edge_file = edges
            self._read_edge_file()
        elif type(edges) == pd.DataFrame:
            self.edge_df = gt.remove_colons(edges).copy()
        self._validate_edges()

        # Process the Node and Edge information
        print('Processing node and edge data...')
        self._process_nodes()
        self._process_edges()

        # Initalize the metagraph and determine the metapaths available
        self._make_metagraph()
        self._determine_metapaths(start_kind, end_kind, max_length)
        self._map_metanodes_to_metaedges()

        # Generate the adjacency matrices.
        print('Generating adjacency matrices...')
        time.sleep(0.5)
        self._generate_adjacency_matrices()

        # Make Degree Weighted matrices.
        print('\nDetermining degrees for each node and metaedge'.format(w))
        time.sleep(0.5)
        self._compute_node_degrees()

        # Make Degree Weighted matrices.
        print('\nWeighting matrices by degree with dampening factor {}...'.format(w))
        time.sleep(0.5)
        self._generate_weighted_matrices()

    def _validate_nodes(self):
        assert 'id' in self.node_df.columns
        assert 'name' in self.node_df.columns
        assert 'label' in self.node_df.columns

    def _validate_edges(self):
        assert 'start_id' in self.edge_df.columns
        assert 'end_id' in self.edge_df.columns
        assert 'type' in self.edge_df.columns

    def _read_node_file(self):
        """Reads the nodes file and stores as a DataFrame."""
        self.node_df = gt.remove_colons(pd.read_csv(self.node_file, dtype=str))

    def _read_edge_file(self):
        """Reads the edge file and stores it as a DataFrame"""
        self.edge_df = gt.remove_colons(pd.read_csv(self.edge_file, dtype=str))

    def _process_nodes(self):
        """Process the nodes DataFrame to generate needed class variables"""
        # Sort the nodes by metanode type, then by id
        self.node_df = self.node_df.sort_values(['label', 'id']).reset_index(drop=True)
        # Get all the ids
        self.nodes = self.node_df['id']
        # Get mapping from the index to the node ID (one to many so need different one for each node type)
        self.index_to_nid = dict()
        for group_name, group in self.node_df.groupby('label'):
            self.index_to_nid[group_name] = group['id'].reset_index(drop=True).to_dict()
        # Get the reverse mapping (many to one so don't need to separate based on type).
        self.nid_to_index = dict()
        for mapper in self.index_to_nid.values():
            for index, nid in mapper.items():
                self.nid_to_index[nid] = index
        # Finally, we need a mapper from id to node type
        self.id_to_metanode = self.node_df.set_index('id')['label'].to_dict()
        # And from node type to a list of ids
        self.metanode_to_ids = dict()
        for group_name, group in self.node_df.groupby('label'):
            self.metanode_to_ids[group_name] = group['id'].tolist()
        # One more mapper of id to name
        self.nid_to_name = self.node_df.set_index('id')['name'].to_dict()

    def _process_edges(self):
        """ Processes the edges to ensure all needed variables are present"""
        # remove any duplicated edges that may linger.
        self.edge_df = self.edge_df.dropna(subset=['start_id', 'end_id', 'type'])
        # Split the metaedge name from its abbreviation if both are included
        if all(self.edge_df['type'].str.contains('_')):
            e_types = self.edge_df['type'].unique()
            e_types_split = [e.split('_') for e in e_types]
            self.metaedges = [e[-1] for e in e_types_split]

            edge_abbrev_dict = {e: abv for e, abv in zip(e_types, self.metaedges)}
            self.edge_df['abbrev'] = self.edge_df['type'].apply(lambda t: edge_abbrev_dict[t])

        # %%TODO Ideally all edges should have their abbreviations included, so this should never be run...
        else:
            self.metaedges = self.edge_df['type'].unique()
            self.edge_df['abbrev'] = self.edge_df['type']

    def _make_metagraph(self):
        """Generates class variable metagraph, an instance of hetio.hetnet.MetaGraph"""

        print('Initializing metagraph...')
        time.sleep(0.5)
        abbrev_dict, edge_tuples = gt.get_abbrev_dict_and_edge_tuples(self.node_df, self.edge_df)
        self.metagraph = MetaGraph.from_edge_tuples(edge_tuples, abbrev_dict)

    def _determine_metapaths(self, start_kind, end_kind, max_length):
        """Generates the class variable metapaths, which has information on each of the meatpaths to be extracted"""

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

        if max_length > 4:
            print('Warning - Max length > 4 is still highly experimental')
            print('Currently only 1 metanode repate is allowed in this mode')
            to_remove = []
            for mp, info in self.metapaths.items():
                if info['length'] <= 4:
                    continue
                else:
                    # Determine if this metapath  metanode the above printed repeat criteria
                    repeats = mt.find_repeated_node_indices(info['edges'])
                    if repeats and (len(repeats) > 1 or len(list(repeats.values())[0]) > 1):
                        to_remove.append(mp)
            for r in to_remove:
                self.metapaths.pop(r)

    def _map_metanodes_to_metaedges(self):
        """ Generate a mapping from Metanode to Metaedges and determine if metanode is at start or end of edge"""
        # look through all metanodes
        for kind in self.metanode_to_ids.keys():

            # Nodes are abbreviated in Metaedge Abbrevioations
            n_abbrev = self.metagraph.kind_to_abbrev[kind]

            metanode_edges = dict()
            # Find out which Metanodes this Metaedge participates in
            for e in self.metaedges:
                parsed = gt.parse_edge_abbrev(e)
                if n_abbrev in parsed:
                    # want to know if our metaedge of interest is the start or end of this metanode...
                    metanode_edges[e] = {'start': parsed[0] == n_abbrev}
            self.metanode_to_edges[kind] = metanode_edges


    def _prepare_args_for_adj_matrix(self, metaedge):
        """
        Create a sparse adjacency matrix for the given metaedge.

        :param metaedge: String, the abbreviation for the metadge. e.g. 'CbG' for Combound-binds-Gene
        :param directed: bool, if the edge is directed (only important for edges between the same metanode).

        :return: Sparse matrix, adjacency matrix for the given metaedge. Row and column indices correspond
            to the order of the node ids in the global variable `nodes`.
        """

        # Subeset the edges based on the metaedge
        edge = self.edge_df.query('abbrev == @metaedge')

        # Find the type and dimensions of the nodes that make up the metaedge
        node_0 = self.id_to_metanode[edge['start_id'].iloc[0]]
        node_1 = self.id_to_metanode[edge['end_id'].iloc[0]]
        dim_0 = self.node_df.query('label == @node_0').shape[0]
        dim_1 = self.node_df.query('label == @node_1').shape[0]

        # Find the start and end nodes for edges of the given type
        start = edge['start_id'].apply(lambda i: self.nid_to_index[i])
        end = edge['end_id'].apply(lambda i: self.nid_to_index[i])

        directed = '>' in metaedge or '<' in metaedge
        homogeneous = node_0 == node_1
        args = {'dim_0': dim_0, 'dim_1': dim_1, 'start': start, 'end': end,
                'directed': directed, 'homogeneous': homogeneous}
        return args

    def _generate_adjacency_matrices(self):
        """Generates adjacency matrices for performing path and walk count operations."""
        self.adj_matrices = dict()
        mes = []
        args = []
        for metaedge in self.metaedges:
            mes.append(metaedge)
            args.append(self._prepare_args_for_adj_matrix(metaedge))
        res = parallel_process(array=args, function=mt.get_adj_matrix, use_kwargs=True, n_jobs=self.n_jobs,
                               front_num=0)
        for metaedge, matrix in zip(mes, res):
            self.adj_matrices[metaedge] = matrix


    def _compute_node_degrees(self):
        """Computes node degree for all nodes and edge types."""
        mes = []
        args = []
        for metaedge, matrix in self.adj_matrices.items():
            mes.append(metaedge)
            args.append(matrix)
        res = parallel_process(array=args, function=mt.calculate_degrees, n_jobs=self.n_jobs, front_num=0)
        for metaedge, (out_degree, in_degree) in zip(mes, res):
            self.out_degree[metaedge] = out_degree
            self.in_degree[metaedge] = in_degree


    def _generate_weighted_matrices(self):
        """Generates the weighted matrices for DWPC and DWWC calculation"""
        self.degree_weighted_matrices = dict()
        mes = []
        args = []
        for metaedge, matrix in self.adj_matrices.items():
            mes.append(metaedge)
            args.append({'matrix': matrix, 'w': self.w, 'degree_fwd': self.out_degree[metaedge],
                         'degree_rev': self.in_degree[metaedge]})
        res = parallel_process(array=args, function=mt.weight_by_degree, use_kwargs=True, n_jobs=self.n_jobs,
                               front_num=0)
        for metaedge, matrix in zip(mes, res):
            self.degree_weighted_matrices[metaedge] = matrix

    def _validate_ids(self, ids):
        """Internal function to ensure that a given id is either a Node type or a list of node ids."""
        if type(ids) == str:
            return self.metanode_to_ids[ids]
        elif isinstance(ids, collections.Iterable):
            # String, assume ids
            if type(ids[0]) == str or type(ids[0]) == np.str_:
                # Ensure all ids belong to metanodes of the same type
                assert len(set([self.id_to_metanode[i] for i in ids])) == 1
                # Sort the ids according to their index in the adj. mats.
                return sorted(ids, key=lambda i: self.nid_to_index[i])

        raise ValueError()


    def update_w(w):
        print('Changing w from {} to {}. Please wait...'.format(self.w, w))

        self.w = w
        self._generate_weighted_matricies()

    def get_node_degree(self, node_id):
        """
        Returns a dictionary of the Edge Types and Corresponding Degrees for the requested node_id.

        :param node_id: string, the identifier for the node

        :return: Dict, key is the abbervation for the edge type, value is the degree of that node
        """
        kind = self.id_to_metanode[node_id]
        idx = self.nid_to_index[node_id]
        node_degrees = dict()

        for metaedge, start in self.metanode_to_edges[kind].items():
            current_matrix = self.adj_matrices[metaedge]
            if start['start']:
                deg = self.out_degree[metaedge][idx]
            else:
                deg = self.in_degree[metaedge][idx]
            node_degrees[metaedge] = deg
        return node_degrees


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

        # If no start or end nodes are passsed, use the original ones used at class formation
        if start_nodes is None:
            start_nodes = self.start_kind
        if end_nodes is None:
            end_nodes = self.end_kind

        # Validate that we either have a list of nodeids, a list of indices, or a string of metanode
        start_ids = self._validate_ids(start_nodes)
        end_ids = self._validate_ids(end_nodes)

        # Get metanode types for start and end
        start_type = self.id_to_metanode[start_ids[0]]
        end_type = self.id_to_metanode[end_ids[0]]

        # Get the ids and names for the start and end to initialize DataFrame
        start_idxs = [self.nid_to_index[i] for i in start_ids]
        end_idxs = [self.nid_to_index[i] for i in end_ids]
        start_name = start_type.lower() + '_id'
        end_name = end_type.lower() + '_id'

        return start_idxs, end_idxs, start_type, end_type, start_ids, end_ids, start_name, end_name

    def _process_extraction_results(self, result, metapaths, start_ids, end_ids, start_name, end_name,
                                    return_sparse=False):
        """
        Internal function to process lists of matrices and metapaths into a DataFrame (or SparseDataFrame) with
        start_node and end_node ids as columns as well as a column for each metapath. Each row will represent the
        path or walk count for a given start_node, end_node pair.
        """
        from itertools import product

        if return_sparse:
            # Turn each result matrix into a series
            print('\nReshaping Result Matrices...')
            time.sleep(0.5)

            size = result[0].shape[0]*result[0].shape[1]
            result = [mt.reshape(res, (size, 1)) for res in tqdm(result)]
            print('Stacking columns...')
            result = hstack(result)
            result = pd.SparseDataFrame(result, columns=metapaths, default_fill_value=0.0)

            # Past all the series together into a DataFrame
            print('\nGenerating DataFrame...')
            start_end_df = pd.DataFrame(list(product(start_ids, end_ids)), columns=[start_name, end_name])
            start_end_df = start_end_df.to_sparse()

            return pd.concat([start_end_df, result], axis=1)

        # Turn each result matrix into a series
        print('\nFormatting results to series...')
        time.sleep(0.5)

        # Currently running in series.  Extensive testing has found no incense in speed via Parallel processing
        # However, parallel usually results in an inaccurate counter.
        for i in tqdm(range(len(metapaths))):
            result[i] = mt.to_series(result[i], name=metapaths[i]).reset_index(drop=True)

            # Past all the series together into a DataFrame
        print('\nConcatenating series to DataFrame...')
        start_end_df = pd.DataFrame(list(product(start_ids, end_ids)), columns=[start_name, end_name])

        return pd.concat([start_end_df] + result, axis=1)

    def _get_parallel_arguments(self, metapaths, start_idxs, end_idxs, start_type,
                                end_type, matrices, verbose, walks=False):
        """Gets the arguments needed for parallel processing"""
        mats_subset_start, mats_subset_end = self._subset_matrices(matrices, start_idxs,
                                                                   end_idxs, start_type, end_type)

        # Prepare functions for parallel processing
        arguments = []
        for mp in metapaths:
            to_multiply = mt.get_matrices_to_multiply(mp, self.metapaths,
                                                      matrices, mats_subset_start, mats_subset_end)
            if not walks:
                edges = mt.get_edge_names(mp, self.metapaths)
                arguments.append({'edges': edges, 'to_multiply': to_multiply,
                                  'start_idxs': start_idxs, 'end_idxs': end_idxs, 'verbose': verbose})
            else:
                arguments.append({'to_multiply': to_multiply})
        return arguments

    def _subset_matrices(self, matrices, start_idxs, end_idxs, start_type, end_type):
        """Subsets starting and ending matrices in a metapath, reducing computational complexity"""
        def get_subset(matrix, idxs, transpose=False):
            mat = matrix.copy()
            if transpose:
                mat = mat.T
            mat = mat.tocsr()
            mt.csr_rows_set_nz_to_val(mat, idxs)
            if transpose:
                mat = mat.T
            return mat

        num_start = len(self.metanode_to_ids[start_type])
        num_end = len(self.metanode_to_ids[end_type])

        leave_out_start = [i for i in range(num_start) if i not in start_idxs]
        leave_out_end = [i for i in range(num_end) if i not in end_idxs]

        mats_subset_start = {}
        mats_subset_end = {}

        edges = list(self.metagraph.get_edges())

        for edge in edges:
            e = edge.get_abbrev()
            if edge.get_id()[0] == start_type:
                mats_subset_start[e] = get_subset(matrices[e], leave_out_start, False)
                if '>' in e and edge.get_id()[1] == start_type:
                    mats_subset_start[mt.get_reverse_directed_edge(e)] = get_subset(matrices[e].T,
                                                                                    leave_out_start, False)
            elif edge.get_id()[1] == start_type:
                mats_subset_start[e] = get_subset(matrices[e], leave_out_start, True)

            if edge.get_id()[1] == end_type:
                mats_subset_end[e] = get_subset(matrices[e], leave_out_end, True)
                if '>' in e and edge.get_id()[0] == end_type:
                    mats_subset_end[mt.get_reverse_directed_edge(e)] = get_subset(matrices[e].T,
                                                                                  leave_out_end, True)
            elif edge.get_id()[0] == end_type:
                mats_subset_end[e] = get_subset(matrices[e], leave_out_end, False)

        return mats_subset_start, mats_subset_end

    def _extract_metapath_feaures(self, metapaths=None, start_nodes=None, end_nodes=None, verbose=False, n_jobs=1,
                                  return_sparse=False, walks=False, degree_weighted=True, message=''):
        """Internal function for extracting any metapath based feature"""
        # Validate the given nodes and get information on nodes needed for results formatting.
        start_idxs, end_idxs, start_type, end_type, start_ids, end_ids, start_name, end_name = \
            self.prep_node_info_for_extraction(start_nodes, end_nodes)

        # Get all metapaths if none passed
        if not metapaths:
            metapaths = sorted(list(self.metapaths.keys()))

        # Choose the correct matrix type for the given feature
        if degree_weighted:
            mats = self.degree_weighted_matrices
        else:
            mats = self.adj_matrices

        # Prepare functions for parallel processing
        print('Preparing function arguments...')
        arguments = self._get_parallel_arguments(metapaths=metapaths, matrices=mats,
                                                 start_idxs=start_idxs, end_idxs=end_idxs, start_type=start_type,
                                                 end_type=end_type, verbose=verbose, walks=walks)

        print('Calculating {}s...'.format(message))
        time.sleep(0.5)

        # Walk and path counts use different functions
        if walks:
            func = mt.count_walks
        else:
            func = mt.count_paths

        result = parallel_process(array=arguments, function=func, use_kwargs=True, n_jobs=n_jobs, front_num=0)

        # Run the feature extraction calculation processes in parallel
        del arguments

        # Process and return results
        results = self._process_extraction_results(result, metapaths, start_ids, end_ids, start_name, end_name,
                                                   return_sparse=return_sparse)
        return results

    def extract_paths(self, start_node, end_node, metapaths=None, degree_weighted=True, n_jobs=1):
        # Validate the given nodes and get information on nodes needed for processing arguments.
        # IDs could potentially be strings or integers..?
        assert type(start_node) == str or not isinstance(start_node, collections.Iterable)
        assert type(end_node) == str or not isinstance(end_node, collections.Iterable)
        assert self.id_to_metanode[start_node] == self.start_kind
        assert self.id_to_metanode[end_node] == self.end_kind
        start_idx = self.nid_to_index[start_node]
        end_idx = self.nid_to_index[end_node]

        # Get all metapaths if none passed
        if not metapaths:
            metapaths = sorted(list(self.metapaths.keys()))
        # If single metapath passed, make it a list
        if type(metapaths) == str:
            metapaths = [metapaths]

        # Choose the correct matrix type for the given feature
        if degree_weighted:
            mats = self.degree_weighted_matrices
        else:
            mats = self.adj_matrices

        arguments = []
        path_nodes = {}
        for mp in metapaths:
            to_multiply = mt.get_matrices_to_multiply(mp, self.metapaths, mats)
            to_multiply[0] = to_multiply[0][start_idx, :]
            to_multiply[len(to_multiply)-1] = to_multiply[-1][:, end_idx]
            arguments.append({'to_multiply': to_multiply, 'start_idx': start_idx,
                              'end_idx': end_idx, 'metapath': mp})
            edges = mt.get_edge_names(mp, self.metapaths)
            path_nodes[mp] = [b.replace('<', '-').replace('>', '-').split(' - ')[0] for b in edges] + \
                                [edges[-1].replace('<', '-').replace('>', '-').split(' - ')[-1]]

        result = parallel_process(array=arguments, function=mt.get_individual_paths,
                                  use_kwargs=True, n_jobs=n_jobs, front_num=0)

        out = []

        for r in result:
            for res in r:
                node_ids = []
                nodes = []
                for idx, node_type in zip(res['node_idxs'], path_nodes[res['metapath']]):
                    node_id = self.index_to_nid[node_type][idx]
                    node = self.nid_to_name[node_id]

                    node_ids.append(node_id)
                    nodes.append(node)

                if len(node_ids) == len(set(node_ids)):
                    out.append({'node_ids': node_ids, 'nodes': nodes, 'metapath': res['metapath'],
                                'metric': res['metric']})

        return out



    def extract_dwpc(self, metapaths=None, start_nodes=None, end_nodes=None, verbose=False, n_jobs=1,
                     return_sparse=False):
        """
        Extracts DWPC metrics for the given metapaths.  If no metapaths are given, will calcualte for all metapaths.

        :param metapaths: list or None, the metapaths paths to calculate DWPC values for.  List must be a subset of
            those found in metapahts.json.  If None, will calcualte DWPC values for metapaths in the metapaths.json
            file.
        :param start_nodes: String or list, String title of the metanode start of the metapaths.
            If a list, can be IDs corresponding to a subset of starting nodes for the DWPC.
        :param end_nodes: String or list, String title of the metanode for the end of the metapaths.  If a
            list, can be IDs corresponding to a subset of ending nodes for the DWPC.
        :param verbose: boolean, if True, prints debugging text for calculating each DWPC. (not optimized for
            parallel processing).
        :param n_jobs: int, the number of jobs to use for parallel processing.
        :param return_sparse: boolean, if true, returns a pandas.SparseDataFrame output.  Good if the data size is
            known to be potentially very large.

        :return: pandas.DataFrame, Table of results with columns corresponding to DWPC values from start_id to
            end_id for each metapath.
        """

        return self._extract_metapath_feaures(metapaths=metapaths, start_nodes=start_nodes, end_nodes=end_nodes,
                                              verbose=verbose, n_jobs=n_jobs, return_sparse=return_sparse,
                                              walks=False, degree_weighted=True, message='DWPC')

    def extract_dwwc(self, metapaths=None, start_nodes=None, end_nodes=None, verbose=False, n_jobs=1,
                     return_sparse=False):
        """
        Extracts DWWC metrics for the given metapaths.  If no metapaths are given, will calcualte for all metapaths.

        :param metapaths: list or None, the metapaths paths to calculate DWPC values for.  List must be a subset of
            those found in metapahts.json.  If None, will calcualte DWPC values for all metapaths in the
            metapaths.json file.
        :param start_nodes: String or list, String title of the metanode for the start of the metapaths.
            If a list, can be IDs corresponding to a subset of starting nodes for the DWWC.
        :param end_nodes: String or list, String title of the metanode for the end of the metapaths.  If a
            list, can be IDs corresponding to a subset of ending nodes for the DWWC.
        :param verbose: boolean, if True, prints debugging text for calculating each DWPC. (not optimized for
            parallel processing).
        :param n_jobs: int, the number of jobs to use for parallel processing.
        :param return_sparse: boolean, if true, returns a pandas.SparseDataFrame output.  Good if the data size is
            known to be potentially very large.

        :return: pandas.DataFrame. Table of results with columns corresponding to DWPC values from start_id to
            end_id for each metapath.
        """

        return self._extract_metapath_feaures(metapaths=metapaths, start_nodes=start_nodes, end_nodes=end_nodes,
                                              verbose=verbose, n_jobs=n_jobs, return_sparse=return_sparse,
                                              walks=True, degree_weighted=True, message='DWWC')

    def extract_path_count(self, metapaths=None, start_nodes=None, end_nodes=None, verbose=False, n_jobs=1,
                           return_sparse=False):
        """
        Extracts path counts for the given metapaths.  If no metapaths are given, will calcualte for all metapaths.

        :param metapaths: list or None, the metapaths paths to calculate DWPC values for.  List must be a subset of
            those found in metapahts.json.  If None, will calcualte DWPC values for metapaths in the metapaths.json
            file.
        :param start_nodes: String or list, String title of the metanode start of the metapaths.
            If a list, can be IDs corresponding to a subset of starting nodes for the DWPC.
        :param end_nodes: String or list, String title of the metanode for the end of the metapaths.  If a
            list, can be IDs corresponding to a subset of ending nodes for the DWPC.
        :param verbose: boolean, if True, prints debugging text for calculating each DWPC. (not optimized for
            parallel processing).
        :param n_jobs: int, the number of jobs to use for parallel processing.
        :param return_sparse: boolean, if true, returns a pandas.SparseDataFrame output.  Good if the data size is
            known to be potentially very large.

        :return: pandas.DataFrame, Table of results with columns corresponding to DWPC values from start_id to
            end_id for each metapath.
        """

        return self._extract_metapath_feaures(metapaths=metapaths, start_nodes=start_nodes, end_nodes=end_nodes,
                                              verbose=verbose, n_jobs=n_jobs, return_sparse=return_sparse,
                                              walks=False, degree_weighted=False, message='Path Count')

    def extract_degrees(self, start_nodes=None, end_nodes=None, subset=None):
        """
        Extracts degree features from the metagraph

        :param start_nodes: string or list, title of the metanode (string) from which paths originate, or a list of
            IDs corresponding to a subset of starting nodes.
        :param end_nodes: string or list, title of the metanode (string) at which paths terminate, or a list of
            IDs corresponding to a subset of ending nodes.
        :param subset: list, the metaedges to extract degrees for if not all are to be extracted.
        :return: DataFrame, with degree features for the given start and end nodes.
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

                # Extract the Degrees and add results to DataFrame
                if start_abbrev:
                    # Need to transpose the correct matrix
                    if start_abbrev == abbrev:
                        degrees = (self.adj_matrices[abbrev] * self.adj_matrices[abbrev].T).diagonal()[start_idxs]
                    else:
                        degrees = (self.adj_matrices[abbrev].T * self.adj_matrices[abbrev]).diagonal()[start_idxs]

                    # Format results to a series
                    start_series = pd.Series(degrees, index=start_ids, dtype='int64')
                    result = result.reset_index()
                    result = result.set_index(start_name)
                    result[start_abbrev] = start_series

                if end_abbrev:
                    if end_abbrev == abbrev:
                        degrees = (self.adj_matrices[abbrev] * self.adj_matrices[abbrev].T).diagonal()[end_idxs]
                    else:
                        degrees = (self.adj_matrices[abbrev].T * self.adj_matrices[abbrev]).diagonal()[end_idxs]

                    # Format to series
                    end_series = pd.Series(degrees, index=end_ids, dtype='int64')
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
        :param start_nodes: String or list, if string, the Metanode for the start of the edge, if list, the ids
            to of nodes to have prior extracted for
        :param end_nodes: String or list, if string, the Metanode for the start of the edge, if list, the ids
            to of nodes to have prior extracted for

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

        # Extract the source and target information from the metagraph
        mg_edge = self.metagraph.metapath_from_abbrev(edge).edges[0]
        if start_nodes is None:
            start_nodes = mg_edge.source.get_id()
        if end_nodes is None:
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

    def is_self_referential(self, edge):
        """Determines if an edge is self-referential"""
        # Determine if edge is directed or not to choose the proper splitting character
        split_str = gt.determine_split_string(edge)

        # split the edge
        edge_split = edge.split(split_str)

        return edge_split[0] == edge_split[-1] and (edge_split[0] == self.start_kind or
                                                    edge_split[0] == self.end_kind)

    def contains_self_referential(self, edges):
        return sum([self.is_self_referential(e) for e in edges]) > 0

    def duplicated_edge_source_or_target(self, edges):
        prev_edge = ['', '', '']
        for edge in edges:
            split_str = gt.determine_split_string(edge)
            edge_split = edge.split(split_str)

            # Ensure start and end on same edge, and same edge semmantics
            if prev_edge[0] == edge_split[-1] and prev_edge[1] == edge_split[1]:
                if prev_edge[0] == self.start_kind or prev_edge[0] == self.end_kind:
                    return True

            prev_edge = edge_split

        return False

    def generate_blacklist(self, target_edge):
        """
        Generates a list of blacklisted features to be excluded from the model

        :param target_edge: str, the name of the target edge to be learned in the model.
        :return: list, the blacklisted features.
        """
        def contains_target(std_abbrevs):
            return target_edge in std_abbrevs

        def is_target(edge):
            return edge == target_edge

        blacklist = []

        # Get degree feature blacklists
        target = [e for e in self.metagraph.get_edges() if e.get_abbrev() == target_edge][0]
        reverse_target = target.inverse.get_abbrev()

        blacklist.append('degree_'+target_edge)
        blacklist.append('degree_'+reverse_target)

        # Get the metapaths features to be blacklisted
        for mp, info in self.metapaths.items():
            num_target = sum([is_target(e) for e in info['standard_edge_abbreviations']])
            num_self_refs = sum([self.is_self_referential(e) for e in info['edges']])

            # Remove edges with overuse of target edge.
            if num_target > 1:
                blacklist.append('dwpc_' + mp)

            # Remove those with 2 self-referential edges and a target edge
            elif num_self_refs > 1 and num_target > 0:
                blacklist.append('dwpc_' + mp)

            # Remove those with a self-reverntal edge, travel across the same edge, and a target:
            elif self.contains_self_referential(info['edges']) \
                    and contains_target(info['standard_edge_abbreviations']) \
                    and self.duplicated_edge_source_or_target(info['edges']):
                blacklist.append('dwpc_' + mp)

        return blacklist
