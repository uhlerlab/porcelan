# Supplementary Data

This folder contains gene sets selected by PORCELAN, GO-term analysis results, 
and lineage trees with edge weights optimized by PORCELAN. 
For _C. elegans_ we analyzed the AB and non-AB lineage separately and ran PORCELAN 
a second time after removing the top weighted genes from the first run to 
select a 2nd gene set for each case. For the mouse embryo samples and a tumor 
sample we ran PORCELAN on 5 different random subsets (distinguishable by the
random seeds in the file names). 

* `gene_sets_selected` contains the gene sets selected by PORCELAN.
* `go_term_analysis` contains adjusted p-values for GO-terms from "GO_Biological_Process_2023" 
   for mouse tumor and embryo samples and "GO_Biological_Process_2018" for _C. elegans_ for 
   the gene sets selected by PORCELAN. 
*  `weighted_trees` contains lineage trees with edge weights optimized by PORCELAN used
   for visualizations in the figures. There are versions obtained by optimizing both
   gene and edge weights jointly and by just optimizing edge weights.
