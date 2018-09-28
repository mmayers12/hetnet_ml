import random
import pandas as pd
from collections import OrderedDict


def get_direction_from_abbrev(abbrev):
    """Finds the direction of a metaedge from its abbreviaton"""
    if '>' in abbrev:
        return 'forward'
    elif '<' in abbrev:
        return 'backward'
    else:
        return 'both'


def get_edge_name(edge):
    """Separates the edge name from its abbreviation"""
    # the true edge name is everything before the final '_' character
    # so if we have PROCESS_OF_PpoP, we still want to keep 'PROCESS_OF' with the underscores intact.
    return '_'.join(edge.split('_')[:-1])


def map_id_to_value(nodes, value):
    """Maps Node id to another value"""
    return remove_colons(nodes).set_index('id')[value].to_dict()


def get_abbrev_dict_and_edge_tuples(nodes, edges):
    """
    Returns an abbreviation dictionary generated from class variables.
    Required input for metagraph functions in the hetio package.

    Edge types are formatted as such:
        edge-name_{START_NODE_ABBREV}{edge_abbrev}{END_NODE_ABBREV}
        e.g. Compound-binds-Gene is: binds_CbG

    Therefore, abbreviations for edge and node types can be extracted from the full edge name.
    """
    nodes = remove_colons(nodes)
    edges = remove_colons(edges)

    id_to_kind = nodes.set_index('id')['label'].to_dict()

    node_kinds = nodes['label'].unique()
    edge_kinds = edges['type'].unique()

    # If we have a lot of edges, lets reduce to one of each type for faster queries later.
    edge_kinds_df = edges.drop_duplicates(subset=['type'])

    # Extract just the abbreviation portion
    edge_abbrevs = [e.split('_')[-1] for e in edge_kinds]

    # Initialize the abbreviation dict (key = fullname, value = abbreviation)
    node_abbrev_dict = dict()
    edge_abbrev_dict = dict()
    metaedge_tuples = []

    for i, kind in enumerate(edge_kinds):
        edge_name = get_edge_name(kind)

        # initialize the abbreviations
        edge_abbrev = ''
        start_abbrev = ''
        end_abbrev = ''

        start = True
        for char in edge_abbrevs[i]:
            # Direction is not in abbreviations, skip to next character
            if char == '>' or char == '<':
                continue

            # When the abbreviation is in uppercase, abbreviating for node type
            if char == char.upper():
                if start:
                    start_abbrev += char
                else:
                    end_abbrev += char

            # When abbreviation is lowercase, you have the abbreviation for the edge
            if char == char.lower():
                # now no longer on the start nodetype, so set to false
                start = False
                edge_abbrev += char

        # Have proper edge abbreviation
        edge_abbrev_dict[edge_name] = edge_abbrev

        # Have abbreviations, but need to get corresponding types for start and end nodes
        edge = edge_kinds_df.query('type == @kind').iloc[0]
        start_kind = id_to_kind[edge['start_id']]
        end_kind = id_to_kind[edge['end_id']]

        node_abbrev_dict[start_kind] = start_abbrev
        node_abbrev_dict[end_kind] = end_abbrev

        direction = get_direction_from_abbrev(kind)
        edge_tuple = (start_kind, end_kind, edge_name, direction)
        metaedge_tuples.append(edge_tuple)

    return {**node_abbrev_dict, **edge_abbrev_dict}, metaedge_tuples


def combine_nodes_and_edges(nodes, edges):
    """Combines data from nodes and edges frames into a single dataframe"""

    nodes = remove_colons(nodes)
    edges = remove_colons(edges)

    id_to_name = map_id_to_value(nodes, 'name')
    id_to_label = map_id_to_value(nodes, 'label')

    out_df = edges.copy()

    out_df['start_name'] = out_df['start_id'].apply(lambda i: id_to_name[i])
    out_df['end_name'] = out_df['end_id'].apply(lambda i: id_to_name[i])

    out_df['start_label'] = out_df['start_id'].apply(lambda i: id_to_label[i])
    out_df['end_label'] = out_df['end_id'].apply(lambda i: id_to_label[i])

    return out_df


def get_node_degrees(edges):
    """Determines the degrees for all nodes"""
    return pd.concat([remove_colons(edges)['start_id'], remove_colons(edges)['end_id']]).value_counts()


def add_colons(df, id_name='', col_types={}):
    """
    Adds the colons to column names before neo4j import (presumably removed by `remove_colons` to make queryable).
    User can also specify  a name for the ':ID' column and data types for property columns.

    :param df: DataFrame, the neo4j import data without colons in it (e.g. to make it queryable).
    :param id_name: String, name for the id property.  If importing a CSV into neo4j without this property,
                    Neo4j mayuse its own internal id's losing this property.
    :param col_types: dict, data types for other columns in the form of column_name:data_type
    :return: DataFrame, with neo4j compatible column headings
    """
    reserved_cols = ['id', 'label', 'start_id', 'end_id', 'type']

    # Get the reserved column names that need to be changed
    to_change = [c for c in df.columns if c.lower() in reserved_cols]
    if not to_change:
        raise ValueError("Neo4j Reserved columns (['id', 'label' 'start_id', 'end_id', 'type'] not " +
                         "found in DataFrame")

    # Add any column names that need to be types
    to_change += [c for c in df.columns if c in col_types.keys()]

    change_dict = {}
    for name in to_change:
        # Reserved column names go after the colon
        if name.lower() in reserved_cols:
            if name.lower() == 'id':
                new_name = id_name + ':' + name.upper()
            else:
                new_name = ':' + name.upper()
        else:
            # Data types go after the colon, while names go before.
            new_name = name + ':' + col_types[name].upper()
        change_dict.update({name: new_name})

    return df.rename(columns=change_dict)


def remove_colons(df):
    """
    Removes colons from column headers formatted for neo4j import to make them queryable

    :param df: DataFrame, formatted for neo4j import (column lables ':ID', ':LABEL, 'name:STRING' etc).
    :return: DataFrame, with column names that are queryable (e.g. 'id', 'label', 'name').
    """
    # Figure out which columns have : in them
    to_change = [c for c in df.columns if ':' in str(c)]
    new_labels = [c.lower().split(':') for c in to_change]

    # keep the reserved types, or names
    reserved_cols = ['id', 'label', 'start_id', 'end_id', 'type']
    new_labels = [l[1] if l[1] in reserved_cols else l[0] for l in new_labels]

    # return the DataFrame with the new column headers
    change_dict = {k: v for k, v in zip(to_change, new_labels)}
    return df.rename(columns=change_dict)


def determine_split_string(edge):
    if '-' in edge:
        return ' - '
    elif '>' in edge:
        return ' > '
    elif '<' in edge:
        return ' < '


def permute_edges(edges, directed=False, multiplier=10, excluded_edges=None, seed=0):
    """
    Permutes the edges of one metaedge in a graph while preserving the degree of each node.

    :param edges: DataFrame, edges information
    :param directed: bool, whether or not the edge is directed
    :param multiplier: int, governs the number of permutations, multiplied by number of edges
    :param excluded_edges: DataFrame, edges to exclude from final permuted edges
    :param seed: int, random state for analysis

    :return permuted_edges, stats: DataFrame, DataFrame - the permuted start and end ids, the permutation stats.
    """
    random.seed(seed)

    orig_columns = edges.columns
    edges = remove_colons(edges)
    col_name_mapper = {k: v for k, v in zip(edges.columns, orig_columns)}

    # There shouldn't be any duplicate edges in the grpah, but throw error just in case
    assert len(edges) == len(edges.drop_duplicates(subset=['start_id', 'end_id']))

    # Ensure only 1 edge type was passed
    assert edges['type'].nunique() == 1
    e_type = edges['type'].unique()[0]

    edge_list = [(e.start_id, e.end_id) for e in edges.itertuples(index=False)]
    edge_set = set(edge_list)
    orig_edge_set = edge_set.copy()

    if excluded_edges is not None:
        excluded_edge_set = set([(e.start_id, e.end_id) for e in excluded_edges.itertuples(index=False)])
    else:
        excluded_edge_set = set()

    edge_number = len(edges)
    n_perm = int(edge_number * multiplier)

    # Initialize some perumtation stats
    count_self_loop = 0
    count_duplicate = 0
    count_undir_dup = 0
    count_excluded = 0

    step = max(1, n_perm // 10)
    print_at = list(range(step, n_perm, step)) + [n_perm - 1]

    stats = list()

    for i in range(n_perm):

        # Same two random edges without replacement
        i_0 = random.randrange(edge_number)
        i_1 = i_0
        while i_0 == i_1:
            i_1 = random.randrange(edge_number)

        edge_0 = edge_list[i_0]
        edge_1 = edge_list[i_1]

        unaltered_edges = [edge_0, edge_1]
        swapped_edges = [(edge_0[0], edge_1[1]), (edge_1[0], edge_0[1])]

        # Validate the new paring
        valid = False
        for edge in swapped_edges:
            # Self Loops
            if edge[0] == edge[1]:
                count_self_loop += 1
                break
                # Duplicate Edges
            if edge in edge_set:
                count_duplicate += 1
                break
                # Duplicate Undirected Edges
            if not directed and (edge[1], edge[0]) in edge_set:
                count_undir_dup += 1
                break
                # Edge is excluded
            if edge in excluded_edge_set:
                count_excluded += 1
                break
                # If we made it here, we have a valid edge
        else:
            valid = True

        # If BOTH new edges are valid
        if valid:

            # Change the edge list
            edge_list[i_0] = swapped_edges[0]
            edge_list[i_1] = swapped_edges[1]

            # Fix the sets for quick hashing
            for edge in unaltered_edges:
                edge_set.remove(edge)
            for edge in swapped_edges:
                edge_set.add(edge)

        if i in print_at:
            stat = OrderedDict()
            stat['cumulative_attempts'] = i
            index = print_at.index(i)
            stat['attempts'] = print_at[index] + 1 if index == 0 else print_at[index] - print_at[index - 1]
            stat['complete'] = (i + 1) / n_perm
            stat['unchanged'] = len(orig_edge_set & edge_set) / len(edges)
            stat['self_loop'] = count_self_loop / stat['attempts']
            stat['duplicate'] = count_duplicate / stat['attempts']
            stat['undirected_duplicate'] = count_undir_dup / stat['attempts']
            stat['excluded'] = count_excluded / stat['attempts']
            stats.append(stat)

            count_self_loop = 0
            count_duplicate = 0
            count_undir_dup = 0
            count_excluded = 0

    assert len(edge_list) == edge_number
    out_edges = pd.DataFrame({'start_id': [edge[0] for edge in edge_list],
                              'end_id': [edge[1] for edge in edge_list],
                              'type': [e_type] * edge_number})

    out_edges = out_edges.rename(columns=col_name_mapper)

    return out_edges, pd.DataFrame(stats)


def permute_graph(edges, multiplier=10, excluded_edges=None, seed=0):
    """
    Permutes the all of the metaedges types for those given in a graph file.

    :param edges: DataFrame, the edges to be permuted
    :param multiplier: int, governs the number of permutations to be performed
    :param excluded_edges: DataFrame, edges to be disallowed from final permutations
    :param seed: int, random state for analysis for reproduciability

    :return permuted_graph, stats: DataFrame, DataFrame - the edges of the graph permuted,
                                   stats on the permutations.
    """
    # Change columns names to pandas standard
    orig_columns = edges.columns
    edges = remove_colons(edges)
    col_name_mapper = {k: v for k, v in zip(edges.columns, orig_columns)}

    edge_types = edges['type'].unique()

    edge_stats = []
    permuted_edges = []
    for i, etype in enumerate(edge_types):
        to_permute = edges.query('type == @etype').copy()

        directed = '>' in etype or '<' in etype
        pedge, stats = permute_edges(to_permute, directed=directed, multiplier=multiplier,
                                     excluded_edges=excluded_edges, seed=seed + len(to_permute))

        permuted_edges.append(pedge)

        stats['etype'] = etype
        edge_stats.append(stats)

    stats = pd.concat(edge_stats)
    permuted_graph = pd.concat(permuted_edges)

    # Return column names to neo4j standards if applicable
    permuted_graph = permuted_graph.rename(columns=col_name_mapper)

    return permuted_graph, stats
