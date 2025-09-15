# AutoDFT
Batch automated DFT and optional vibronic spectrum calculation with FCclasses3. Continuously a work-in-progress. No guarantees of accuracy, completeness, or functionality.

Written in Python (3.11.13).


## Highlights and main functions
* Requires only an input ChemDraw structure for the molecule of interest
* Automated 3D geometry building and coarse minimization with OpenBabel
* Generates a linked job file for Gaussian16, containing:
  * Rough geometry optimization (HF/STO-3G)
  * Opt+freq DFT (B3LYP/6-31G(d,p))
  * Opt+freq TD-DFT (B3LYP/6-31G(d,p))
* Automated vibronic spectrum calculation with FCclasses3
* Deuteration support (by replacing a specific atom)


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

### Optional Python modules (useful for eventual analysis)
* pandas
* matplotlib
* re
* numpy
* openpyxl

Note: the code is currently written assuming Gaussian16 is the available CCP, but it should be readily adapted to Orca, Qchem, etc.

## First-time setup
Ensure that the ```$PATH``` and ```$LD_LIBRARY_PATH``` are set appropriately for OpenBabel and FCclasses (point to ```bin/``` and ```lib64/``` folders), and that any requisite libraries/modules are loaded before running.

In the ```auto_DFT.py``` Python code, you'll also have to update the paths to point to where OpenBabel, FCclasses, Gaussian, etc. are installed on your machine.

Make sure all Python dependencies are installed:

```
> pip install pandas matplotlib numpy argparse openpyxl
```


## Organization
The main code (```auto_DFT.py```) can be called in a folder organized as:

```
Folder/
├── auto_DFT.py
├── full_FCC_batch.py
└── Subfolder1/
    └── Molecule1.cdx
└── Subfolder2/
    └── Molecule2.cdx
└── Subfolder3/
    └── Molecule3.cdx
```

## Input structures
The inputs _must_ be ```.cdx``` files. ```.cdxml``` files are not currently supported.

Be sure that the structures don't have any abbreviated functional groups. For example, a sulfate group can't be abbreviated as -SO3H. The _full connectivity_ must be drawn out.

No isotopic labels or other unique labels are currently supported. For deuterium labeling, use the ```--deuterate``` option (described below).

## Usage
Calling ```python auto_DFT_v#.py``` will run the code. 

There are also several optional arguments:
* --cores : specify the amount of processor cores available
* --mem : specify the amount of RAM available
* --jobtype : specify the calculations to be performed ("raman", "gsonly", or ground+excited state (default))
* --deuterate : atoms to be replaced by deuterium in the final calculation (e.g., place fluorine atoms at the desired deuterium locations and use ```--deuterate F```)

```
> python auto_DFT_v#.py --cores 28 --mem 100 --deuterate F --jobtype raman
```

For submitting to a cluster, a sample SLURM job file is provided (batch_dft.sh).



# How to cite
If you found any of these functions useful, please consider citing this code as:

1. PA Kocheril et al. (in preparation).
