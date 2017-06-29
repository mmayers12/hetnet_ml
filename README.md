# hetnet-ml

Software to quickly extract features from heterogenous networks for machine learning.

## Usage

    In [1]: from extractor import MatrixFormattedGraph
    
    In [2]: mg = MatrixFormattedGraph('nodes.csv', 'edges.csv', 'metapaths.json')
    Reading file information...
    Generating adjcency matrices...
    100%|███████████████████████████████████████████████████████████████| 24/24 [01:23<00:00,  7.51s/it]
    Weighting matrices by degree with dampening factor 0.4...
    100%|███████████████████████████████████████████████████████████████| 25/25 [00:31<00:00,  1.26s/it]
    
    In [3]: res = mg.calculate_dwpc(start_nodes='Compound', end_nodes='Disease')
    Calculating DWPCs...
    100%|███████████████████████████████████████████████████████████| 1206/1206 [22:34<00:00,  1.12s/it]
    Reformating results...
        
