# Data pre-processing

For convenience we include all small pre-processed data files
needed to reproduce the paper figures in the `preprocessed` 
directory. These files can also be reproduced from the files
provided in the original datasets using the corresponding
jupyter notebooks provided here.

Notes about a few special cases:
* `3435_NT_T1_apn_pd_triplet_lut.npy` is not included due to GitHub
  file size restrictions. This file can easily be reproduced by
  running the corresponding cells in `tumor_preprocessing.ipynb`.
* Similarly, `GSE117542_embryo2_normalized_log_counts.txt` and
  `GSM3302829_embryo3_normalized_log_counts.txt` are not included but
  can be reproduced by running the corresponding cells in
  `mouse_embryo_preprocessing.ipynb`
* The lineage trees published with the mouse embryogenis dataset
  from from Chan et al. are missing cell lables. We used additional
  files that we received to from the authors to resolve the labels
  and applied our pre-processing to create the pre-processed lineage
  trees provided in the `preprocessed` directory.
