import pandas as pd

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
    return nodes.set_index(':ID')[value].to_dict()


def get_abbrev_dict_and_edge_tuples(nodes, edges):
    """
    Returns an abbreviation dictionary generated from class variables.

    Edge types are formatted as such:
        edge-name_{START_NODE_ABBREV}{edge_abbrev}{END_NODE_ABBREV}
        e.g. Compound-binds-Gene is: binds_CbG

    Therefore, abbreviations for edge and node types can be extracted from the full edge name.
    """

    id_to_kind = nodes.set_index(':ID')[':LABEL'].to_dict()

    node_kinds = nodes[':LABEL'].unique()
    edge_kinds = edges[':TYPE'].unique()

    # If we have a lot of edges, lets reduce to one of each type for faster queries later.
    edge_kinds_df = edges.drop_duplicates(subset=[':TYPE'])

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
        edge = edge_kinds_df[edge_kinds_df[':TYPE'] == kind].iloc[0]
        start_kind = id_to_kind[edge[':START_ID']]
        end_kind = id_to_kind[edge[':END_ID']]

        node_abbrev_dict[start_kind] = start_abbrev
        node_abbrev_dict[end_kind] = end_abbrev

        direction = get_direction_from_abbrev(kind)
        edge_tuple = (start_kind, end_kind, edge_name, direction)
        metaedge_tuples.append(edge_tuple)

    return {**node_abbrev_dict, **edge_abbrev_dict}, metaedge_tuples


def combine_nodes_and_edges(nodes, edges):
    """Combines the nodes and edges frames into a single dataframe for simple analysis"""

    id_to_name = map_id_to_value(nodes, 'name')
    id_to_label = map_id_to_value(nodes, ':LABEL')

    out_df = edges.copy()

    out_df['start_name'] = out_df[':START_ID'].apply(lambda i: id_to_name[i])
    out_df['end_name'] = out_df[':END_ID'].apply(lambda i: id_to_name[i])

    out_df['start_label'] = out_df[':START_ID'].apply(lambda i: id_to_label[i])
    out_df['end_label'] = out_df[':END_ID'].apply(lambda i: id_to_label[i])

    return out_df


def get_node_degrees(nodes):
    """Determines the degrees for all nodes"""
    return pd.concat([nodes[':START_ID'], nodes[':END_ID']]).value_counts()


def add_colons(df):
    """Adds colons to column names required for Neo4j import"""
    if 'LABEL' in df.columns:
        return df.rename(columns={'ID': ':ID', 'LABEL': ':LABEL'})
    elif 'TYPE' in df.columns:
        return df.rename(columns={'START_ID': ':START_ID', 'END_ID': ':END_ID', 'TYPE': ':TYPE'})
    raise ValueError('LABEL or TYPE not in the columns')


def remove_colons(df):
    """Removes colons from column labels to make them queryable"""
    new_labels = [c[1:] for c in df.columns if str(c).startswith(':')]
    old_labels = [c for c in df.columns if str(c).startswith(':')]
    change_dict = {k: v for k, v in zip(old_labels, new_labels)}

    return df.rename(columns=change_dict)
