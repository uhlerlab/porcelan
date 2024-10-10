# Data pre-processing

The pre-processed data files can be reproduced from the files
provided in the original datasets using the corresponding
jupyter notebooks provided here. If you want to skip this step, 
you can also download them [here](https://drive.google.com/file/d/1LsZ-fRpiVB4-OGRJc4btmioj9PDevYJy/view?usp=sharing) 
and add them to the `preprocessed` directory. 

A few special case:
* The lineage trees published with the mouse embryogenis dataset
  from from Chan et al. are missing cell lables. We used additional
  files that we received to from the authors to resolve the labels
  for embryo 2 and the lineage tree from the [HotSpot tutorial](
  https://hotspot.readthedocs.io/en/latest/Lineage_Tutorial.html) for
  embryo 3 and provided the labeled trees directly in the `mouse` directory.
* Not all cells in the C. elegans dataset from Packer et al. had 
  cell type annotations. We filled in the missing types based on
  this [list](https://www.wormatlas.org/celllistsulston.htm) from Sulston 
  et al. and provide a table of them in the `worm` directory. 
