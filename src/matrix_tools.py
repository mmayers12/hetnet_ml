import numpy as np
import pandas as pd

from scipy.sparse import diags, eye, csc_matrix, csr_matrix, coo_matrix
from collections import defaultdict
from itertools import combinations, chain


def get_path(metapath, metapaths):
    """
    Finds the correct abbreviations and order for the metaedges in a metapath.

    :param metapath: String, the abbreviation for the metapath e.g. 'CbGaD'
    :param metapaths: dict, with keys metapaths, and values dicts containing metapath information including
        edge_abbreviations and standard_edge_abbreviations.

    :return: list, contains the abbreviations for each metaedge in the metapath.
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
    :param metapath: String, the abbreviation for the metapath e.g. 'CbGaD'
    :param metapaths: dict, with keys metapaths, and values dicts containing metapath information including
        edge_abbreviations and standard_edge_abbreviations.
    :return: list, the full names of each of the edges in the metapath
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
    backward = False
    if len(orig_spl) == 1:
        orig_spl = orig.split('<')
        backward = True

    if not backward:
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

        # This is some ugly code... It puts the end node in the start position,
        # Adds a '<' then the edge abbreviation,
        # then grabs the start node abbreviation by indices and puts it in the end position.
        return orig_spl[1] + '<' + orig[edge[0]: edge[-1] + 1] + orig[start_node[0]: start_node[-1] + 1]
    else:
        # The metanode is in upper(), whereas the metaedge is in lower()
        # Use this to find the correct indices regarding the node and edge
        orig1 = orig_spl[1].lower()
        orig2 = orig_spl[1].upper()

        start_node = []
        edge = []

        # Add indices for the start node name (orig != orig.lower())
        for i, (l0, l1) in enumerate(zip(orig_spl[1], orig1)):
            if l0 != l1:
                start_node.append(len(orig_spl[0]) + 1 + i)

        # Add indices for the edge abbreviation (orig != orig.upper())
        for i, (l0, l2) in enumerate(zip(orig_spl[1], orig2)):
            if l0 != l2:
                edge.append(len(orig_spl[0]) + 1 + i)

        # This is some ugly code... It puts the end node in the start position,
        # Adds a '<' then the edge abbreviation,
        # then grabs the start node abbreviation by indices and puts it in the end position.
        return orig[start_node[0]: start_node[-1] + 1] + orig[edge[0]: edge[-1] + 1] + '>' + orig_spl[0]


def weight_by_degree(matrix, w=0.4):
    """
    Weights an adjacency matrix by node degree.

    :param matrix: The sparse matrix to be weighted
    :param w: Dampening factor for weighting the edges. 0 < w <= 1  Default = 0.4

    :return: Sparse Adjacency Matrix, weighted by degree
    """

    degree_fwd = (matrix * matrix.T).diagonal()
    degree_rev = (matrix.T * matrix).diagonal()

    # set 0s to 1s
    degree_fwd[np.where(degree_fwd == 0)] = 1
    degree_rev[np.where(degree_rev == 0)] = 1

    # Weight each degree
    weighted_degree_fwd = degree_fwd ** (-1 * w)
    weighted_degree_rev = degree_rev ** (-1 * w)

    matrix_out = matrix.T.multiply(weighted_degree_fwd)
    matrix_out = matrix_out.T.multiply(weighted_degree_rev)

    return matrix_out.astype('float32').tocsc()


def count_walks(to_multiply):
    """
    Calculates either WC or DWWC depending on wither an adj. or weighted matrix is passed. Walks essentially
    allow repeated visits to the same node, whereas paths do not.

    :param to_multiply: The list of sparse matrices to be used to calculate.  If a simple adjacency matrix,
        will give Walk Count, but if a matrices are weighted by degree will give Degree Weighted Walk Count

    :return: The matrix giving the number of walks, where matrix[i, j], the ith node is the starting node and the
        jth node is the ending node
    """
    return np.prod(to_multiply)


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
            index_pairs.append([indices[i], indices[i + 1]])
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
    size = matrices[0].shape[1]
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
            inner_product = [m for m in to_multiply[start: end] if
                             m.shape[0] != size or m.shape[1] != size or (m != eye(size)).sum()]
            inner_product = count_between_identical_metanodes(inner_product)
            to_multiply[start] = inner_product
            for i in range(start + 1, end):
                to_multiply[i] = eye(size)

    # Remove identity matrices from list before final multiplication
    to_multiply = [m for m in to_multiply if m.shape[0] != size or m.shape[1] != size or (m != eye(size)).sum()]

    return np.prod(to_multiply)


def count_paths_removing_repeated_type(path, edges, matrices, repeat_type, default_to_max=False):
    """
    Counts paths removing repeats due to only one repeated metanode in the metapath.

    A String or List can be passed as the repeat type.  A String will only look for repates in
    that metanode type, while a list will look for all types in the list, chosing the first metanode type,
    in list order, identified as being repeated. A flag to default to the metanode type with the most repeats
    can be used if no given types are found.

    :param path: list, the standard edge abbreviations that make up the metapath
    :param edges: list, the edge names that make up the metapath
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


def estimate_count_from_repeats(edges, to_multiply, resolving_function=interpolate_overcounting):
    """
    Estimates the Path-Count based on the differnece between the Walk-Count and the Path-Count removing
    repated nodes for each metanode type individually.

    Finds the extra counts for each metanode type that is visited more than once, then uses the
    resolving_function to estimate the overcounting via walks.  If this value ends up being greater
    than the total walk-count, the Path-Count is set to zero.

    :param edges: list, the edge names that make up the metapath
    :param to_multiply: List of the matrices to use for calculation
        (e.g. degree weighted or standard adjacency)
    :param resolving_function: function to determine the error.  Default: sum()
    """

    # Initialize values
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


def calc_abab(mats, return_steps=False):
    """
    Counts paths with an ABAB structure. Takes a list of 3 matrices, removes overcounts due to visiting A twice,
    removes overcounts due to visiting B twice, then adds back in paths that were doubly removed where an A and B
    node were both visited twice.

    :param mats: list, the matrcies to be multiplied together
    :param return_steps: Boolean, if True, will also return the intermediate steps, for further calculations
    :return: Matrix, the path counts.
    """
    assert len(mats) == 3

    step1 = mats[0] * mats[1]
    step2 = mats[1] * mats[2]

    diag1 = diags(step1.diagonal())
    diag2 = diags(step2.diagonal())

    overcount1 = diag1 * mats[2]
    overcount2 = mats[0] * diag2

    doubly_removed = mats[0].multiply(mats[1].T.multiply(mats[2]))

    result = np.prod(mats) - overcount1 - overcount2 + doubly_removed

    # ensure we didn't subtract to produce negative counts
    result[result < 0] = 0

    if return_steps:
        return result, step1, step2

    return result


def get_abab_list(to_multiply, all_repeats):
    """
    Gets a list of matrices that conform to ABAB pattern. Collapses down larger patterns like ABCAB to ABAB.

    :param to_multiply: list, the matrices to be multiplied to determine the path count.
    :param all_repeats: list, the locations of the repeats.

    :return: list of len 3 that makes the ABAB pattern.
    """
    abab_list = []

    start = all_repeats[0][0]
    end = all_repeats[1][0]

    abab_list.append(np.prod(to_multiply[start:end]))

    start = all_repeats[1][0]
    end = all_repeats[0][1]

    abab_list.append(np.prod(to_multiply[start:end]))

    start = all_repeats[0][1]
    end = all_repeats[1][1]

    abab_list.append(np.prod(to_multiply[start:end]))
    return abab_list


def determine_abab_kind(repeat_indices, to_multiply):
    """
    Determines the ABAB structure in the metapath and selects the appropriate path counting method.

    :param repeat_indices: list, the locations where nodes are repeated in the path structure.
    :param to_multiply: list, the matrices to be multiplied together to get the path count.

    :return: Matrix, the path counts
    """
    all_repeats = sorted(list(c for c in chain(*repeat_indices)))
    if len(all_repeats) == 2:

        abab_list = get_abab_list(to_multiply, all_repeats)
        abab_result = calc_abab(abab_list)

        if min(list(chain(*all_repeats))) != 0:
            return to_multiply[0] * abab_result
        elif max(list(chain(*all_repeats))) != len(to_multiply):
            return abab_result * to_multiply[-1]
        else:
            return abab_result

    elif len(all_repeats) == 3:
        return abab_3(repeat_indices, to_multiply)


def abab_3(repeat_indices, to_multiply):
    """
    Determines the path counts for a path with ABAB structure and either node A or B is repeated a 3rd time.

    :param repeat_indices: list, the locations where nodes are repeated in the path structure.
    :param to_multiply: list, the matrices to be multiplied together to get the path count.

    :return: Matrix, the path counts
    """
    ordered_repeats = sorted(repeat_indices, key=lambda x: len(x))
    shorter = ordered_repeats[0][0]
    longer = ordered_repeats[1]
    if min(shorter) > max(longer[0]):
        # Comes before ABAB (e.g. AABAB)
        result, step1, step2 = calc_abab(to_multiply[1:], True)
        overcount3 = diags((to_multiply[0] * step1).diagonal()) * to_multiply[-1]
        result = (to_multiply[0] * result) - overcount3
    elif max(shorter) < min(longer[1]):
        # comes after ABAB (e.g ABABB)
        result, step1, step2 = calc_abab(to_multiply[:-1], True)
        overcount3 = to_multiply[0] * diags((step2 * to_multiply[-1]).diagonal())
        result = (result * to_multiply[-1]) - overcount3
    else:
        return None
    # ensure we didn't subtract to produce negative counts
    result[result < 0] = 0
    return result


def expand_matrix(matrix, size, start_idxs=None, end_idxs=None):
    """
    Expands a matrix a sub-setted square adjcency matrix on start_idxs and/or end_idxs, back to a square shape.

    :param matrix: scipy.sparse matrix that has been subsetted
    :param size: int, the original size of the matrix
    :param start_idxs: list, the row indices that the original matrix was sub-setted on
    :param end_idxs: list, the column indices that the original mtrix was sub-setted on

    :return: scipy.sparse.csc_matrix, size x size in dimension.
    """

    def h_expand(in_mat, idxs):

        # Create the square output matrix.
        out_mat = np.zeros((1, size))[0]
        out_mat = diags(out_mat).tolil()

        # Subset the number of rows if the in matrix isn't already fully expanded on rows.
        if in_mat.shape[0] < size:
            out_mat = out_mat[:in_mat.shape[0], :]

        # Copy the input matrix values to the correct rows
        for i, idx in enumerate(idxs):
            out_mat[:, idx] = in_mat[:, i]
        return out_mat

    out_matrix = matrix
    # Loop allows for expansion of matrices sub-setted both on start and end indices
    while out_matrix.shape[0] < size or out_matrix.shape[1] < size:

        # Do a vertical expand (rows) if needed
        if start_idxs is not None and out_matrix.shape[0] < size:
            out_matrix = h_expand(out_matrix.tocsr().T, start_idxs).T
        # Do a horizontal expansion (columns) if needed
        if end_idxs is not None and out_matrix.shape[1] < size:
            out_matrix = h_expand(out_matrix, end_idxs)

    return out_matrix.tocsc()


def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row + 1]] = value


def csr_rows_set_nz_to_val(csr, rows, value=0):
    for row in rows:
        csr_row_set_nz_to_val(csr, row)
    if value == 0:
        csr.eliminate_zeros()


def get_matrices_to_multiply(metapath, metapaths, matrices, mats_subset_start=None, mats_subset_end=None):
    """
    Finds the correct abbreviations and order for the metaedges in a metapath.

    :param metapath: String, the abbreviation for the metapath e.g. 'CbGaD'
    :param metapaths: dict, with keys metapaths, and values dicts containing metapath information including
        edge_abbreviations and standard_edge_abbreviations.
    :param matrices: dictionary of matrices from which to generate the path
    :param mats_subset_start: dictionary of matrices from which to generate the path, only to be used for the first
        step in a metapath if and only if some of the starting nodes are not being extracted.
    :param mats_subset_end: dictionary of matrices from which to generate the path, only to be used for the final
        step in a metapath if and only if some of the ending nodes are not being extracted.
    :return: list, the matrices to be multiplied in the path.
    """
    edge_abbrevs = metapaths[metapath]['edge_abbreviations']
    std_edge_abbrevs = metapaths[metapath]['standard_edge_abbreviations']

    # Determine which matrices need to be transposed
    transpose = []
    for ea, std_ea in zip(edge_abbrevs, std_edge_abbrevs):
        if ea != std_ea:
            if '>' not in std_ea:
                transpose.append(True)
            else:
                transpose.append(False)
        else:
            transpose.append(False)

    # Select the matrices to be multiplied and transpose if necessary
    to_multiply = []
    for i, (ea, std_ea, trans) in enumerate(zip(edge_abbrevs, std_edge_abbrevs, transpose)):
        # Use sub-setted start matrices on first iteration
        if i == 0 and mats_subset_start is not None:
            mats = mats_subset_start
        # Use sub-setted end matrices on last iteration
        elif i == len(edge_abbrevs) - 1 and mats_subset_end is not None:
            mats = mats_subset_end
        else:
            mats = matrices

        # Select the correct matrix with correct transpose state for path
        if '>' in ea or ('<' in ea and ea in mats.keys()):
            to_multiply.append(mats[ea])
        elif '<' in ea:
            to_multiply.append(mats[get_reverse_directed_edge(ea)].T)
        elif trans:
            to_multiply.append(mats[std_ea].T)
        else:
            to_multiply.append(mats[std_ea])

    return to_multiply


def count_paths(edges, to_multiply, start_idxs=None, end_idxs=None, verbose=False,
                uncountable_estimate_func=estimate_count_from_repeats, uncountable_params=None):
    """
    Counts paths removing repeats due to only one repeated metanode in the metapath.

    A String or List can be passed as the repeat type.  A String will only look for repeats in
    that metanode type, while a list will look for all types in the list, choosing the first metanode type,
    in list order, identified as being repeated. A flag to default to the metanode type with the most repeats
    can be used if no given types are found.

    :param edges: Dictionary with information on each metapath
    :param to_multiply: list of matrices to multiply for the calculation
        (e.g. degree weighted or standard adjacency)
    :param start_idxs: list of ints, the indices of the starting nodes in the original square matrix
    :param end_idxs: list of ints, the indices of the ending nodes in the original square matrix
    :param verbose: boolean, if True, prints results of decision tree logic.
    :param uncountable_estimate_func: Function to determine the path count when matrix multiplication cannot return
        an exact answer. Must be a function `metapath` and `matricies`. Any other parameters can be passed by
        the `uncountable_params` argument.
    :param uncountable_params: Dictionary, the keyword arguments for any other parameters to be passed to the
        uncountable_estimate_func.

    :return: Sparse Matrix, containing path counts along the metapath.
    """

    if start_idxs is None:
        start_idxs = np.arange(to_multiply[0].shape[0])
    if end_idxs is None:
        end_idxs = np.arange(to_multiply[-1].shape[1])

    repeated_nodes = find_repeated_node_indices(edges)

    # uncountable params must be a dict.
    if not uncountable_params:
        uncountable_params = {}

    # Nothing repeated, so walks == paths, just need product
    if not repeated_nodes:
        if verbose:
            print('No repeats')
        return np.prod(to_multiply)[start_idxs, :][:, end_idxs]

    # Only 1 metanode type is repeated, so easy to get exact answer
    elif len(repeated_nodes) == 1:
        if verbose:
            print('1 repeat...', end=' ')
        repeat_indices = list(repeated_nodes.values())
        if len(repeat_indices[0]) > 2:
            if verbose:
                print('4 Visits... Estimating')
                return uncountable_estimate_func(edges, to_multiply, **uncountable_params)[start_idxs, :][:, end_idxs]

        if verbose:
            print('Countable')
        return count_removing_repeats(repeat_indices, to_multiply)[start_idxs, :][:, end_idxs]

    # 2 repeated metanode types, Fast to determine if exact answer is determinable
    elif len(repeated_nodes) == 2:
        if verbose:
            print('2 repeats...', end=' ')

        repeats = sorted(list(repeated_nodes.values()))

        if is_countable(repeats):
            if verbose:
                print('Countable')
            # if Countable, repeats that start second will always come before those
            # that start first.
            repeat_indices = sorted(repeats, reverse=True)
            return count_removing_repeats(repeat_indices, to_multiply)[start_idxs, :][:, end_idxs]

        else:
            if verbose:
                print('trying new ABAB logic...', end=' ')

            result = determine_abab_kind(repeats, to_multiply)
            if result is not None:
                if verbose:
                    print('Success')
                return result[start_idxs, :][:, end_idxs]

            else:
                if verbose:
                    print('Estimating')
                return uncountable_estimate_func(edges, to_multiply, **uncountable_params)[start_idxs, :][:, end_idxs]

    elif len(repeated_nodes) > 2:
        print('Not yet implemented', edges)
        print('Returning Zeroes')
        return csc_matrix(np.zeros((len(start_idxs), len(end_idxs))))

    else:
        print("Unknown error, Something went wrong.....", edges)
        print("Returning Zeroes")
        return csc_matrix(np.zeros((len(start_idxs), len(end_idxs))))


def get_individual_paths(to_multiply, start_idx, end_idx, metapath):
    out = []

    if len(to_multiply) == 2:
        result = to_multiply[0].multiply(to_multiply[1].T)
        row_nz = result.nonzero()[0]
        col_nz = result.nonzero()[1]

        for r, c in zip(row_nz, col_nz):
            node_idxs = []
            node_idxs.append(start_idx)
            node_idxs.append(c)
            node_idxs.append(end_idx)

            out.append({'node_idxs': node_idxs, 'metric': result[r, c], 'metapath': metapath})


    elif len(to_multiply) == 3:
        result = to_multiply[0].T.multiply(to_multiply[1]).multiply(to_multiply[2].T)
        row_nz = result.nonzero()[0]
        col_nz = result.nonzero()[1]

        for r, c in zip(row_nz, col_nz):
            node_idxs = []
            node_idxs.append(start_idx)
            node_idxs.append(r)
            node_idxs.append(c)
            node_idxs.append(end_idx)

            out.append({'node_idxs': node_idxs, 'metric': result[r, c], 'metapath': metapath})


    elif len(to_multiply) == 4:
        result = [0]*to_multiply[1].shape[0]
        first_res = to_multiply[0].T.multiply(to_multiply[1])

        for row in first_res.sum(axis=1).nonzero()[0]:

            result[row] = first_res[row, :].T.multiply(to_multiply[2]).multiply(to_multiply[3].T)

            nz_rows = result[row].nonzero()[0]
            nz_cols = result[row].nonzero()[1]

            for r, c in zip(nz_rows, nz_cols):
                node_idxs = []
                node_idxs.append(start_idx)
                node_idxs.append(row)
                node_idxs.append(r)
                node_idxs.append(c)
                node_idxs.append(end_idx)

                out.append({'node_idxs': node_idxs, 'metric': result[row][r, c], 'metapath': metapath})
    return out


def reshape(a, shape):
    """Reshape the sparse matrix `a`.

    Returns a coo_matrix with shape `shape`.
    """
    if not hasattr(shape, '__len__') or len(shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')

    c = a.tocoo()
    nrows, ncols = c.shape
    size = nrows * ncols

    new_size =  shape[0] * shape[1]
    if new_size != size:
        raise ValueError('total size of new array must be unchanged')

    flat_indices = ncols * c.row + c.col
    new_row, new_col = divmod(flat_indices, shape[1])

    b = coo_matrix((c.data, (new_row, new_col)), shape=shape)
    return b.tocsc()


def to_series(result, start_ids=None, end_ids=None, name=None):
    """
    Convert a result matrix (containing pc, dwpc, degree values) to a Series with multiindex start_id, end_id.

    :param result: Sparse matrix containing the caluclation's result.
    :param start_ids: list of ids corresponding to the start of the path
    :param end_ids: list of ids corresponding to the end of the path
    :param name: string, name for the returned Series

    :return: pandas.Series, with multi-index start_id, end_ide and values corresponding to the metric calulated.
    """
    dat = pd.DataFrame(result.todense(), index=start_ids, columns=end_ids)

    # Convert to series
    dat = dat.stack()
    dat.name = name

    return dat
