import bz2
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.special import logit
from itertools import combinations
from scipy.sparse import diags, eye
from collections import defaultdict


def get_path(metapath, metapaths):
    """
    Finds the correct abbreviations and order for the metaedges in a metapath.

    :param metapath: String, the abbrevation for the metapath e.g. 'CbGaD'
    :return: list, contains the abbereviations for each metaedge in the metapath.
        e.g. ['CbG', 'GaD']
    """
    # If no directed edges, just return standard abbreviations
    if '<' not in metapath and '>' not in metapath:
        return metapaths[metapath]['standard_edge_abbreviations']

    # Directed edges, so non-standard abbreviations are required for correct direction
    path = metapaths[metapath]['edge_abbreviations']
    out_path = []

    # Find the directed edges and keep non-standard, otherwise take the standard abbrev.
    for i, e in enumerate(path):
        if '<' in e or '>' in e:
            out_path.append(e)
        else:
            out_path.append(metapaths[metapath]['standard_edge_abbreviations'][i])
    return out_path


def get_edge_names(metapath, metapaths):
    """Gets the list of edge names from a metapath abbreviation
    :param metapaths:
    """
    return metapaths[metapath]['edges']


def get_reverse_directed_edge(orig):
    """
    Generates the string corresponding to the reverse metaedge of a directed edge.

    :param orig: String, The abbreviation for the original forward edge e.g. 'Gr>G'
    :return: String, the abbreviation for the reversed edge e.g. 'G<rG'
    """

    # Split at the character '>'
    # Everything after '>' will remain the same
    orig_spl = orig.split('>')

    # The metanode is in upper(), whereas the metaedge is in lower()
    # Use this to find the correct indices regarding the node and edge
    orig1 = orig_spl[0].lower()
    orig2 = orig_spl[0].upper()

    start_node = []
    edge = []

    # Add indices for the start node name (orig != orig.lower())
    for i, (l0, l1) in enumerate(zip(orig_spl[0], orig1)):
        if l0 != l1:
            start_node.append(i)

    # Add indices for the edge abbreviation (orig != orig.upper())
    for i, (l0, l2) in enumerate(zip(orig_spl[0], orig2)):
        if l0 != l2:
            edge.append(i)

    # This is some ugly code... It grabs the start node abbreviation by indices
    # Adds a '<' then the edge abbreviation, then the end node abbreviation
    return orig[start_node[0]: start_node[-1]+1] + '<' + orig[edge[0]: edge[-1]+1] + orig_spl[1]


def get_reverse_undirected_edge(orig):
    """
    Gets the reverse edge for a standard edge.
    :param orig:
    :return:
    """

    nodes = ['', '']
    edge = ''
    idx = 0
    for l0, l1 in zip(test, u):
        if l0 == l1:
            nodes[idx] += l0
        else:
            edge += l0
            idx = 1
    return nodes[1] + edge + nodes[0]


def weight_by_degree(matrix, w=0.4, directed=False):
    """
    Weights an adjacency matrix by node degree.

    :param matrix: The sparse matrix to be weighted
    :param w: Dampening factor for weighting the edges. 0 < w <= 1  Default = 0.4
    :param directed: Boolean, for directed edges. If True, calculates in-degree and out-degree separated.

    :return: Sparse Adjacency Matrix, weighted by degree
    """

    # Get the degree of each item
    if directed:
        # In and out degrees are different in directed edges
        out_degree = matrix.sum(axis=1)
        out_degree = np.array(np.reshape(out_degree, len(out_degree)))[0]
        in_degree = np.array(matrix.sum(axis=0))[0]

        # set 0s to 1 for negative exponents to work
        in_degree[np.where(in_degree == 0)] = 1
        out_degree[np.where(out_degree == 0)] = 1

        # weight degrees
        weighted_in = in_degree ** (-1*w)
        weighted_out = out_degree ** (-1*w)

        # Rows * out_degree
        # Cols * in_degree
        matrix_out = matrix.multiply(weighted_in).transpose()
        matrix_out = matrix_out.multiply(weighted_out).transpose()

    else:
        degree = (matrix*matrix).diagonal()

        # set 0s to 1s
        degree[np.where(degree == 0)] = 1

        # Weight each degree
        weighted_degree = degree ** (-1*w)

        matrix_out = matrix.multiply(weighted_degree).transpose()
        matrix_out = matrix_out.multiply(weighted_degree)   # symmetric matrix, so second transpose unneeded

    # Return weighted edges
    return matrix_out.tocsc()


def count_walks(path, matrices):
    """
    Calculates either WC or DWWC depending on wither an adj. or weighted matrix is passed. Walks essentially
    allow repeated visits to the same node, whereas paths do not.

    :param path: list, the abbreviations for each metaedge of the metapath to be followed.
    :param matrices: The dictionary of sparse matrices to be used to calculate.  If a simple adjacency matrix,
        will give Walk Count, but if a matrices are weighted by degree will give Degree Weighted Walk Count

    :return: The matrix giving the number of walks, where matrix[i, j], the ith node is the starting node and the
        jth node is the ending node
    """
    size = matrices[path[0]].shape[0]

    # initialize result with identity matrix
    result = eye(size)
    # Multiply by each metaedge in the metapath
    for edge in path:
        result *= matrices[edge]
    return result


def find_repeated_node_indices(edge_names):
    """
    Determines which metanodes are visited more than once in a metapath froms the edge names.
    Returns these values as a dictionary with key as metanode and values a list of indices that
    tell the step number along the metapath where that metaedge is visited.

    e.g. CuGuCrCbGaD would return {'Compound': [[0, 2], [2, 3]], 'Gene': [[1, 4]]}

    :param edge_names: List of Strings, the proper names for the edges. e.g. 'Compound - binds - Gene'

    :return: Dictionary, with keys the metanodes that are visited more than once, and values, the index
        where that node is repeatedly visited.
    """

    # Just get the node types visited from the edge names
    visited_nodes = [edge_names[0].split(' ')[0]]
    for e in edge_names:
        visited_nodes.append(e.split(' ')[-1])

    # Convert to index for each node type
    node_order = defaultdict(list)
    for i, n in enumerate(visited_nodes):
        node_order[n].append(i)

    # Remove nodes that are only visited once
    node_order = {k: v for k, v in node_order.items() if len(v) > 1}

    # Reshape into start and stop paris of indices
    # e.g. [0, 2, 3] becomes [[0, 2], [2, 3]]
    for node_type, indices in node_order.items():
        index_pairs = []
        for i in range(len(indices) - 1):
            index_pairs.append([indices[i], indices[i+1]])
        node_order[node_type] = index_pairs

    return node_order


def multiply_removing_diagonal(matrices, only_repeat_paths=False):
    """
    Counts paths between two metanodes of same type along the metaedges given

    :param matrices: List of sparse matrices that make up metaedges to count paths
    :param only_repeat_paths: Boolean, if True, returns only the walks with repeat nodes, due to the diagonal
        formed.

    :return: Sparse Matrix, count of the paths with the walks that visit the same node removed.
        if only_repat_paths is set to True, will only return walks that start and end on the same node.
    """

    # Count all the walks along the metapath
    rough_count = np.prod(matrices)
    # Find those which start and end on the same node
    repeats = diags(rough_count.diagonal())
    if only_repeat_paths:
        return repeats
    # Remove the repeated walks to count only the paths
    return rough_count - repeats


def is_unweighted(matrix):
    """Quick function to determine if a Matrix is likely an unweighted adjacency matrix"""
    return matrix[matrix.nonzero()].min() == 1 and matrix.max() == 1


def count_between_identical_metanodes(path):
    """
    Given a metapath where each metaedge starts and ends on the same metanode, this counts the paths between them.

    :param path: List of sparse matrices representing the edges. Each one must start and end on the same
        metanode

    :return: Sparse matrix, the path counts down the metapath, with repeats removed
    """
    result = multiply_removing_diagonal(path[0])
    if len(path) == 1:
        return result
    else:
        for edge in path[1:]:
            result = multiply_removing_diagonal([result, edge])
    return result


def count_removing_repeats(repeat_indices, matrices):
    """
    Counts paths removing walks with repeated nodes. Repeat indices identify where in the metapath metanodes are
    being revisited.

    Matrices are multiplied in order of repeat indices, so the following can be achieved:
    (a*(b*c)*d)*e can be achieved by passing [[1,2],[0,3]], [a,b,c,d,e] as the fucntions arguments.
    Aftery multiplying b*c, repeated nodes are removed by setting the resultant matrix diagonal to 0.
    Then, that product will be multiplied by a and d, (a*prod*d), and again the result will have its
    diagonal set to 0. Finally that result will be multiplied by e.

    :param repeat_indices: List of lists, pairs of indices showing the start and stop subpaths leading
        to repeated metanodes.  These are to be put in the order which the to_multiply should be multiplied
    :param matrices: List of sparse to_multiply making up the metapath, which are to be multiplied.

    :return: Sparse matrix, containing the counts for each path.
    """

    # Get the matrix size in case identity to_multiply need to be made
    size = matrices[0].shape[0]
    to_multiply = matrices[:]

    # Multiply in the order of the indices
    for indices in repeat_indices:
        for idxs in indices:
            start = idxs[0]
            end = idxs[1]

            # Do the multiplication
            sub_result = multiply_removing_diagonal(to_multiply[start: end])

            # Insert the result into the list
            to_multiply[start] = sub_result

            # Collapse remaining to_multiply from this result
            # With identity matrix
            for i in range(start + 1, end):
                to_multiply[i] = eye(size)

        # If returns to the same metanode multiple times, more cycles need to be removed
        if len(indices) > 1:
            # Multiply through the cycles
            start = np.min(indices)
            end = np.max(indices)

            # Remove the identity matrices and multiply from start to end of the same typed nodes
            inner_product = [m for m in to_multiply[start: end] if (m != eye(size)).sum()]
            inner_product = count_between_identical_metanodes(inner_product)
            to_multiply[start] = inner_product
            for i in range(start + 1, end):
                to_multiply[i] = eye(size)

    # Remove identity matrices from list before final multiplication
    to_multiply = [m for m in to_multiply if (m != eye(size)).sum()]

    return np.prod(to_multiply)


def count_paths_removing_repeated_type(path, edges, matrices, repeat_type, default_to_max=False):
    """
    Counts paths removing repeats due to only one repeated metanode in the metapath.

    A String or List can be passed as the repeat type.  A String will only look for repates in
    that metanode type, while a list will look for all types in the list, chosing the first metanode type,
    in list order, identified as being repeated. A flag to default to the metanode type with the most repeats
    can be used if no given types are found.

    :param path: String representation of metapath to count paths for
    :param matrices: Dictionary of the matrices to use for calculation
        (e.g. degree weighted or standard adjacency)
    :param repeat_type: String or list, the metanode type to remove repeats from.
    :param default_to_max: Boolean, default to the most repeated metanode type if `repeat_type` is not repetead

    :return: Sparse Matrix, containing path counts along the metapath.
    """

    # Initalize values
    to_multiply = [matrices[edge] for edge in path]
    repeated_nodes = find_repeated_node_indices(edges)

    # Find indices for the locations of repeated nodes
    repeat_indices = []

    # If given a list, prioritize by order and see if any of the metanodes are repeated
    if type(repeat_type) == list:
        for kind in repeat_type:
            if kind in repeated_nodes:
                repeat_indices = repeated_nodes[kind]
                break

    # if passed a string, try to access it's indices
    elif type(repeat_type) == str:
        repeat_indices = repeated_nodes.get(repeat_type, [])

    # Didn't find any of the explicitly passed types,
    # If default flag is true, use type with most repeats
    if default_to_max and not repeat_indices:
        max_length = 0
        for val in repeated_nodes.values():
            if len(val) > max_length:
                repeat_indices = val
                max_length = len(val)

    return count_removing_repeats(repeat_indices, to_multiply)


def contains(small, big):
    """Returns True if small is perfectly contained in big in order without interruption"""
    following = big.index(small[0]) + 1
    return big[following] == small[1]


def is_countable(indices):
    """
    Determines if paths are countable by matrix multiplication when given the indices of
    two different repeated metanode types:

    :param indices: List of Lists, containing the start and end indices of
        visites to metanodes of each type along the metapath in question

    :return: Boolean, returns True, if metanode vists of one type are either separate or contained within
        visits of another type, such that repeated paths can be removed via analysis of the diagonal
        when performaing matrix multiplaction.
    """
    countable = []
    indices = sorted(indices)

    # Countable is eihter no overlap between repeat visits of the same type,
    # or perfect overlap of one within another
    for indices1 in indices[0]:
        for indices2 in indices[1]:
            total = sorted(indices1 + indices2)
            countable.append(contains(indices1, total) or contains(indices2, total))

    return all(countable)


def get_elementwise_max(matrices):
    """Gets element-wise maximum from a list of sparse matrices"""

    result = matrices[0]
    for matrix in matrices[1:]:
        result = result.maximum(matrix)
    return result


def interpolate_overcounting(extra_counts):
    """Interpolate to find a good approximation of the error due to visiting multiple nodes twice."""

    # Element-wise max underestimates, summing overestimates, so take the average of those two values
    return (get_elementwise_max(extra_counts) + sum(extra_counts)) / 2


def estimate_count_from_repeats(path, edges, matrices, resolving_function=interpolate_overcounting):
    """
    Estimates the Path-Count based on the differnece between the Walk-Count and the Path-Count removing
    repated nodes for each metanode type individually.

    Finds the extra counts for each metanode type that is visited more than once, then uses the
    resolving_function to estimate the overcounting via walks.  If this value ends up being greater
    than the total walk-count, the Path-Count is set to zero.

    :param path: String representation of metapath to count paths for
    :param matrices: Dictionary of the matrices to use for calculation
        (e.g. degree weighted or standard adjacency)
    :param resolving_function: function to determine the error.  Default: sum()
    """

    # Initialize values
    to_multiply = [matrices[edge] for edge in path]
    repeated_nodes = find_repeated_node_indices(edges)

    repeat_indices = [v for v in repeated_nodes.values()]

    # If there is only one node type, but it is repeated 4 or more times,
    # Group into sets of 3 repeats
    if len(repeat_indices[0]) > 2:
        new_indices = list(combinations(repeat_indices[0], 2))
        repeat_indices = new_indices

    # Find the overestimate via walks
    walks = np.prod(to_multiply)

    extra_counts = []

    # Find the extra paths due to repeats of each node type
    for indices in repeat_indices:
        high_est = count_removing_repeats([indices], to_multiply)
        extra_counts.append(walks - high_est)

    # Resolve the extra counts
    result = walks - resolving_function(extra_counts)
    result[result < 0] = 0

    return result


def count_paths(path, edges, matrices, verbose=False, uncountable_estimate_func=estimate_count_from_repeats,
                uncountable_params=None):
    """
    Counts paths removing repeats due to only one repeated metanode in the metapath.

    A String or List can be passed as the repeat type.  A String will only look for repeats in
    that metanode type, while a list will look for all types in the list, choosing the first metanode type,
    in list order, identified as being repeated. A flag to default to the metanode type with the most repeats
    can be used if no given types are found.

    :param path: String representation of metapath to count paths for
    :param edges: Dictionary with information on each metapath
    :param matrices: Dictionary of the matrices to use for calculation
        (e.g. degree weighted or standard adjacency)
    :param verbose: boolean, if True, prints results of decision tree logic.
    :param uncountable_estimate_func: Function to determine the path count when matrix multiplication cannot return
        an exact answer. Must be a function `metapath` and `matricies`. Any other parameters can be passed by
        the `uncountable_params` argument.
    :param uncountable_params: Dictionary, the keyword arguments for any other parameters to be passed to the
        uncountable_estimate_func.

    :return: Sparse Matrix, containing path counts along the metapath.
    """

    to_multiply = [matrices[edge] for edge in path]
    repeated_nodes = find_repeated_node_indices(edges)

    # uncountable params must be a dict.
    if not uncountable_params:
        uncountable_params = {}

    # Nothing repeated, so walks == paths, just need product
    if not repeated_nodes:
        if verbose:
            print('No repeats')
        return np.prod(to_multiply)

    # Only 1 metanode type is repeated, so easy to get exact answer
    elif len(repeated_nodes) == 1:
        if verbose:
            print('1 repeat')
        repeat_indices = list(repeated_nodes.values())
        if len(repeat_indices[0]) > 2:
            if verbose:
                print('4 Visits, Estimating')
                return uncountable_estimate_func(path, edges, matrices, **uncountable_params)

        return count_removing_repeats(repeat_indices, to_multiply)

    # 2 repeated metanode types, Fast to determine if exact answer is determinable
    elif len(repeated_nodes) == 2:
        if verbose:
            print('2 repeats')

        repeats = sorted(list(repeated_nodes.values()))

        if is_countable(repeats):
            if verbose:
                print('Countable')
            # if Countable, repeats that start second will always come before those
            # that start first.
            repeat_indices = sorted(repeats, reverse=True)
            return count_removing_repeats(repeat_indices, to_multiply)

        else:
            if verbose:
                print('Estimating')
            result = uncountable_estimate_func(path, edges, matrices, **uncountable_params)
            return result

    elif len(repeated_nodes) > 2:
        print('Not yet implemented', path)

    else:
        print("Something went wrong.....", path)


def to_series(result, start_nodes, end_nodes, index_to_id, name=None):
    """
    Convert a result matrix (containing pc, dwpc, degree values) to a Series with multiindex start_id, end_id.

    :param result: Sparse matrix containing the caluclation's result.
    :param start_nodes: list of indices corresponding to the start of the path
    :param end_nodes: list of indices corresponding to the end of the path
    :param index_to_id: dict, to map the indices to an id
    :param name: string, name for the returned Series

    :return: pandas.Series, with multi-index start_id, end_ide and values corresponding to the metric calulated.
    """
    dat = pd.DataFrame(result.todense()[start_nodes, :][:, end_nodes],
                       index=[index_to_id[sid] for sid in start_nodes],
                       columns=[index_to_id[eid] for eid in end_nodes])

    # Convert to series
    series = dat.stack()
    series.name = name

    return series
