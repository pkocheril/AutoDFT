# AutoDFT
Batch automated DFT job creation and submission. Continuously a work-in-progress. No guarantees of accuracy, completeness, or functionality.

Written in Python (3.11.13).


## Highlights and main functions
* Automated 3D geometry building and coarse minimization with OpenBabel
* Generates a linked job file for Gaussian16, containing:
  * Rough geometry optimization (HF/STO-3G)
  * Opt+freq DFT (B3LYP/6-31G(d,p))
  * Opt+freq TD-DFT (B3LYP/6-31G(d,p))


## Prerequisites
* OpenBabel
* Gaussian 16
* Python


### Required Python modules
* os
* subprocess
* Path
* defaultdict
* glob
* argparse
* math

Note: the code is currently written assuming Gaussian16 is the available CCP, but it should be readily adapted to Orca, Qchem, etc.

## Organization
The main code ("auto_DFT_v#") should contain all dependencies and can be called in a folder organized as:

```
Folder/
├── auto_DFT_v#.py
└── Subfolder1/
    └── Molecule1.cdx
└── Subfolder2/
    └── Molecule2.cdx
└── Subfolder3/
    └── Molecule3.cdx
```


## Usage
Calling "python auto_DFT_v#.py" will run the code. For submitting to a cluster, a sample SLURM job file is provided (batch_dft.sh).


# How to cite
If you found any of these functions useful, please consider citing this code as:

1. PA Kocheril et al. (in preparation).
