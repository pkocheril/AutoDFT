# AutoDFT
Batch automated DFT and optional vibronic spectrum calculation with FCclasses3. Continuously a work-in-progress. No guarantees of accuracy, completeness, or functionality.

Written in Python (3.11.13).


## Highlights and main functions
* Requires only an input ChemDraw structure for the molecule of interest
* Automated 3D geometry building and coarse minimization with OpenBabel
* Generates a linked job file for Gaussian16, generally performing:
  * Rough geometry optimization (HF/STO-3G)
  * Opt+freq DFT (B3LYP/6-31G(d,p)/SMD)
  * Opt+freq TD-DFT (B3LYP/6-31G(d,p)/SMD)
* Vibronic spectrum calculation with FCclasses3
* AutoDFT 2.0: Deuteration support (by replacing a specific atom)
* Raman scattering spectrum calculations (including pre-resonance Raman)
* AutoDFT 3.0: Anharmonic calculations specifically for a mode of interest (e.g., nitrile), including resuming from a previous calculation
* Single-script, fully automated spectrum plotting and printout (including Gaussian broadening to resemble real spectra)


## Dependences
* OpenBabel
* Gaussian 16
* FCclasses3
* Python


### Required Python modules
* os
* subprocess
* Path
* defaultdict
* glob
* argparse
* math
* pandas
* matplotlib
* re
* numpy

Note: the code is currently written assuming Gaussian16 is the available CCP, but it should be readily adapted to Orca, Qchem, etc.


## First-time setup
Ensure that the ```$PATH``` and ```$LD_LIBRARY_PATH``` are set appropriately for OpenBabel and FCclasses (point to ```bin/``` and ```lib64/``` folders), and that any requisite libraries/modules (e.g., ```openblas``` or ```mkl```) are loaded before running.

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

No isotopic labels or other unique labels are currently supported in the ```.cdx``` files themselves. For deuterium labeling, use the ```--deuterate``` option (described below).

## Usage
At minimum, calling ```python auto_DFT.py``` will run the code. 

There are also several optional arguments:

* --cores : specify the amount of processor cores available
* --mem : specify the amount of RAM available
* --jobtype : specify the calculations to be performed ("raman", "gsonly", or ground+excited state (default))
* --deuterate : atoms to be replaced by deuterium in the final calculation (e.g., place fluorine atoms at the desired deuterium locations and use ```--deuterate F```)
* --FCtype : specify the type of FCclasses run to be performed
* --preexcite : whether to use vibrational pre-excitation (needed for BonFIRE)
* --Nmodes : the number of modes to pre-excite in (leave unspecified to do all modes)
* --wmin and --wmax: frequency bounds for analysis of a mode of interest
* --scaling : set scaling factor for vibrational frequency calculations (for plotting only)
* --rerunFC : whether to force rerun FCclasses calculations (not supported for OPA)


### Examples
Infrared spectrum calculation:

```
> python auto_DFT.py --cores 28 --mem 100 --jobtype gsonly
```

Raman scattering calculation for a deuterated probe:

```
> python auto_DFT.py --cores 28 --mem 100 --deuterate F --jobtype raman
```

BonFIRE calculation:

```
> python auto_DFT.py --cores 28 --mem 100 --FCtype OPA --preexcite yes --bonfire yes
```

Pre-resonance Raman calculation for a nitrile dye:

```
> python auto_DFT.py --cores 28 --mem 100 --FCtype RR --wmin 2000 --wmax 2400
```

For submitting to a cluster, a sample SLURM job file is provided (batch_dft.sh).


# How to cite
If you found any of these functions useful, please cite this code as:

1. PA Kocheril, RE Leighton, N Naji, D Lee, H Wang, J Du, and L Wei*. Towards accurate predictions of bond-selective fluorescence spectra. DOI: 10.48550/arXiv.2601.11902

Also be sure to properly cite the other packages used in this code (OpenBabel, Gaussian16, FCclasses3, etc.).
