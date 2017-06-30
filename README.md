# hetnet-ml

Software to quickly extract features from heterogenous networks for machine learning.

## Requirments

Requires scipy 0.19.0 or greater

## Usage

    In [1]: from extractor import MatrixFormattedGraph
    
    In [2]: mg = MatrixFormattedGraph('nodes.csv', 'edges.csv', 'metapaths.json')
    Reading file information...
    Generating adjcency matrices...
    100%|███████████████████████████████████████████████████████████████| 24/24 [01:02<00:00,  6.11s/it]
    
    Weighting matrices by degree with dampening factor 0.4...
    100%|███████████████████████████████████████████████████████████████| 25/25 [00:25<00:00,  2.52s/it]
    
    In [3]: res = mg.calculate_dwpc(start_nodes='Compound', end_nodes='Disease', n_jobs=32)
    Calculating DWPCs...
    100%|███████████████████████████████████████████████████████████| 1206/1206 [03:31<00:00,  4.71s/it]
    
    Reformating results...
    100%|███████████████████████████████████████████████████████████| 1206/1206 [03:45<00:00,  4.53it/s]
    
    In [4]: res.head(2)
    Out[4]: 
      compound_id    disease_id  CbGdCuGuD  CuGpCCpGdD  CrCuGaD  CuGeAeGaD  \
    0     DB00014  DOID:0050156          0    0.000968        0   0.000425   
    1     DB00014  DOID:0050425          0    0.000000        0   0.000350   
    
       CtDaGcGaD  CuGuDuGdD  CpDuG<rGuD  CbGbCbGaD     ...       CpDaGaD  \
    0          0   0.000662           0          0     ...             0   
    1          0   0.000000           0          0     ...             0   
    
       CbG<rGcGdD  CdGdDrDrD  CbGuCuGaD  CtDtCrCpD  CuGcGbCtD  CdGiGbCpD  \
    0           0          0          0   0.008822          0          0   
    1           0          0          0   0.001407          0          0   
    
       CdG<rGuCtD  CbGdDaGaD  CuG<rG<rGuD  
    0           0          0     0.001432  
    1           0          0     0.000000  
    
    [2 rows x 1208 columns]

        
    In [5]: res.shape
    Out[5]: (212624, 1208)

    In [6]: import bz2
    
    In [7]: with bz2.open('dwpc-features.tsv.bz2', 'wt') as write_file:
       ...:     res.to_csv(write_file, sep='\t', index=False, float_format='%.4g')


