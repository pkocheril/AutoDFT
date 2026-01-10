#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Fully automated batch DFT and FCclasses for BonFIRE spectrum generation.

Call as: python (this_script).py --cores $(nproc) --mem $(nmem) --deuterate F --jobtype gsonly --FCtype OPA --preexcite yes --Nmodes 1 --wmin 2000 --wmax 2400 --scaling 0.953 --rerunFC no --bonfire yes

Optional arguments:
--cores $(nproc) # number of processor cores
--mem $(nmem) # available memory in GB
--deuterate F # atom to replace with deuterium
--jobtype (gsonly, raman, default) # only do ground-state, Raman, or default (gs+ex for FCclasses)
--FCtype (OPA, MCD, EMI, RR) # type of FCclasses run (default OPA)
--pre-excite (yes, no) # whether to run FCclasses with vibrational pre-excitation (default no)
--Nmodes (number of modes to pre-excite) # only used if pre-excite is yes (default all modes)
--wmin (minimum frequency to extract) (default 2000)
--wmax (maximum frequency to extract) (default 2400)
--scaling (scaling factor for frequencies) # 0.953 for triple bonds, 0.97 for double bonds (default 0.953)
--rerunFC (yes, no) # whether to rerun FCclasses or not (default no)

To-do:
- add FCC IR job option
- break up batch_FCC_run function
"""

# Import modules
import os
import subprocess
from pathlib import Path
from collections import defaultdict
import glob
import argparse
import math
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Custom module
import pybonfire as bf

################## Setup ##################

# User variables
mol_charge = 0 # 0 by default, will update if needed
mol_spin = 1 # singlets by default
cores = 4 # 4 by default, will use argparse to get nproc
ram = 4 # 4 by default, will use argparse to get nmem (custom script)
solvent = "dmso" # implicit solvent to be used
solvent_model = "smd" # definitely use SMD for FCclasses -- see paper for optimization
CALC_TYPE = "EMI" # OPA, EMI, RR, MCD

# Set paths
if os.path.isdir("/resnick/groups/WeiLab/software/"): # if on the cluster
    OBABEL_PATH = "/resnick/groups/WeiLab/software/openbabel/bin/obabel"
    G16_PATH = "/resnick/groups/WeiLab/software/g16/g16"
    FC_CLASSES_PATH = "/resnick/groups/WeiLab/software/fcclasses3-3.0.3/src/main/fcclasses3"
    FORMCHK_PATH = "/resnick/groups/WeiLab/software/g16/formchk"
    GENFCC_PATH = "/resnick/groups/WeiLab/software/fcclasses3-3.0.3/FCCBIN/gen_fcc_state"
    GENDIP_PATH = "/resnick/groups/WeiLab/software/fcclasses3-3.0.3/FCCBIN/gen_fcc_dipfile"
    oncluster = 1 # 1 = true
else: # not on the cluster - won't run DFT calculations
    OBABEL_PATH = "/usr/local/openbabel/bin/obabel"
    G16_PATH = "/Applications/GaussView6/gv/g16"
    FC_CLASSES_PATH = "/usr/local/bin/fcclasses3"
    FORMCHK_PATH = "/Applications/GaussView6/gv/formchk"
    GENFCC_PATH = "/usr/local/bin/gen_fcc_state"
    GENDIP_PATH = "/usr/local/bin/gen_fcc_dipfile"
    oncluster = 0

################## Functions ##################

########## General utility ##########

def run_cmd(cmd, cwd=None):
    "Runs a command-line command."
    
    #print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)

def look_for_files(ext, cwd=None):
    """Searches current or specified directory 
    for files of a specific extension."""
    
    found_files = []
    if cwd:
        found_files = glob.glob(f"{cwd}/*.{ext}")
    else:
        found_files = glob.glob(f"*.{ext}")
    return found_files

def find_unique_file(pattern): # ensure only one file of a given type, used in batch_run_FCC
    matches = glob.glob(pattern)
    if len(matches) == 0:
        #print(f"Error: No file found matching '{pattern}' in the current directory.")
        return None
    elif len(matches) > 1:
        #print(f"Error: Multiple files found matching '{pattern}': {matches}")
        return None
    return matches[0]

def readlines(file_path):
    """Read a file and return the contents."""
    with open(file_path, 'r') as file:
        return file.readlines()

def writelines(file_path, contents):
    """Write contents to a file."""
    with open(file_path, 'w') as file:
        file.writelines(contents)

########## DFT ##########

def make_gjf(mol_path, job_base_name, charge=0, multiplicity=1, cpus=31, memory=190, jobtype=None):
    "Generates a .gjf input for molecule parametrization."
    
    mol_file = Path(mol_path)
    assert mol_file.exists(), f"MOL file not found: {mol_path}"
    
    with mol_file.open() as f:
        mol_lines = f.readlines()

    atom_lines = []
    bonds = defaultdict(list)

    atom_index = 1
    atom_indices = {}  # maps line number to atom index for bonds

    for line in mol_lines:
        parts = line.strip().split()
        if len(parts) >= 16:
            x, y, z = map(float, parts[:3])
            atom = parts[3]
            atom_lines.append(f"{atom:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f}")
            atom_indices[len(atom_lines)] = atom_index
            atom_index += 1
        elif len(parts) == 7 and all(p.isdigit() for p in parts[:3]):
            a1, a2, bond_order = map(int, parts[:3])
            bonds[a1].append((a2, float(bond_order)))

    # Build Gaussian connectivity section
    connectivity_lines = []
    for atom_idx in range(1, len(atom_lines) + 1):
        conn = bonds.get(atom_idx, [])
        if conn:
            line = f"{atom_idx} " + " ".join(f"{i} {b:.1f}" for i, b in conn)
        else:
            line = f"{atom_idx}"
        connectivity_lines.append(line)

    full_input = ""

    # Define all available job types
    all_jobs = {
        "rough_opt": {
            "oldcheck": "",
            "checkpoint": f"{job_base_name}_roughopt",
            "method": "HF",
            "basis": "STO-3G",
            "keywords": "opt geom=connectivity", # can't use empirical dispersion here
            "title": f"{job_base_name} HF/STO-3G rough geometry optimization",
        },
        "ground_state": {
            "oldcheck": f"{job_base_name}_roughopt",
            "checkpoint": f"{job_base_name}_gs",
            "method": "B3LYP",
            "basis": "6-31G(d,p)",
            "keywords": f"opt freq=(noraman,savenormalmodes) nosymm pop=nbo scrf=({solvent_model},solvent={solvent}) geom=allcheck",
            "title": f"{job_base_name} ground-state", # needs to be geom=allcheck to resume calculation
        },
        "excited_state": {
            "oldcheck": f"{job_base_name}_gs",
            "checkpoint": f"{job_base_name}_ex",
            "method": "B3LYP",
            "basis": "6-31G(d,p)",
            "keywords": f"opt freq=(noraman,savenormalmodes) td=singlets nosymm pop=nbo scrf=({solvent_model},solvent={solvent}) geom=allcheck",
            "title": f"{job_base_name} excited-state",
        },
        "raman": {
            "oldcheck": f"{job_base_name}_roughopt",
            "checkpoint": f"{job_base_name}_gs",
            "method": "B3LYP",
            "basis": "6-31G(d,p)",
            "keywords": f"opt freq=(raman,savenormalmodes) nosymm pop=nbo scrf=({solvent_model},solvent={solvent}) geom=allcheck",
            "title": f"{job_base_name} ground-state Raman",
        },
        "anharmall": {
            "oldcheck": f"{job_base_name}_gs",
            "checkpoint": f"{job_base_name}_anharm",
            "method": "B3LYP",
            "basis": "6-31G(d,p)",
            "keywords": f"opt freq=(noraman,savenormalmodes,anharmonic) nosymm pop=nbo scrf=({solvent_model},solvent={solvent}) geom=allcheck",
            "title": f"{job_base_name} anharmonic",
        },
        "anharmCN": {
            "oldcheck": f"{job_base_name}_gs",
            "checkpoint": f"{job_base_name}_anharmCN",
            "method": "B3LYP",
            "basis": "6-31G(d,p)",
            "keywords": f"opt freq=(noraman,savenormalmodes,anharmonic,selectanharmonicmodes) nosymm pop=nbo scrf=({solvent_model},solvent={solvent}) geom=allcheck",
            "title": f"{job_base_name} nitrile anharmonic", # also needs "modes=#" and 2 blank lines at the end
        },
    }

    # Define job type configurations
    job_configs = {
        "gsonly": ["rough_opt", "ground_state"],
        "raman": ["rough_opt", "raman"],
        "default": ["rough_opt", "ground_state", "excited_state"],
        "fromscratch": ["rough_opt", "ground_state", "anharmall"],
        "resumeCN": ["anharmCN"],
    }

    # Select jobs based on jobtype
    selected_job_names = job_configs.get(jobtype, job_configs["default"])
    jobs = [all_jobs[name] for name in selected_job_names]

    for i, job in enumerate(jobs):
        link = "\n--link1--\n" if i > 0 else ""
        chargemult = f"\n{charge} {multiplicity}\n" if i == 0 else ""
        checks = (
            f"%oldchk={job['oldcheck']}.chk\n%chk={job['checkpoint']}.chk"
            if i > 0 else
            f"%chk={job['checkpoint']}.chk"
        )
        route = f"# {job['method']}/{job['basis']} {job['keywords']}"
        block = (
            f"{link}%nprocshared={cpus}\n"
            f"%mem={memory}GB\n"
            f"{checks}\n"
            f"{route}\n\n"
            f"{job['title']}\n"
            f"{chargemult}"
        )

        if i == 0:
            block += "\n".join(atom_lines) + "\n\n" + "\n".join(connectivity_lines) + "\n"

        block += "\n"
        full_input += block

    input_path = mol_file.parent / f"{job_base_name}_linked.gjf"
    input_path.write_text(full_input)
    return input_path

def replace_with_deuteriums(gjf_path, target_string="F     ",replace_string="H(iso=2)"):
    "Replaces target atoms with deuteriums in a Gaussian job file (default is fluorine)."

    with open(gjf_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.startswith(target_string):
            lines[i] = line.replace(target_string, replace_string)
    with open(gjf_path, 'w') as file:
        file.writelines(lines)

########## FCclasses ##########

def batch_run_FCC(n,calctype):
    # Look for fcc input files in current folder
    state1_file = find_unique_file("*_gs*fcc")  # gs file is STATE1_FILE
    state2_file = find_unique_file("*_ex*fcc")  # ex file is STATE2_FILE
    eldip_file = find_unique_file("eldip*fchk")
    magdip_file = find_unique_file("magdip*fchk")

    # Stop execution if any required file is missing or ambiguous
    if not state1_file or not state2_file or not eldip_file:
        # Look for .chk files in current folder
        state1_chk = find_unique_file("*_gs.chk")
        state2_chk = find_unique_file("*_ex.chk")
        
        if not state1_chk or not state2_chk:
            print("Unambiguous ground-state and excited-state files not found – exiting.")
            return
        else:
            # Generate fchk files
            subprocess.run([FORMCHK_PATH, state1_chk], cwd=os.getcwd(), check=True)
            subprocess.run([FORMCHK_PATH, state2_chk], cwd=os.getcwd(), check=True)
            # Look for fchk files
            state1_fchk = find_unique_file("*_gs.fchk")
            state2_fchk = find_unique_file("*_ex.fchk")
            if not state1_fchk or not state2_fchk:
                return
            else:
                # Generate state and dipole files
                subprocess.run([GENFCC_PATH, "-i", state1_fchk], cwd=os.getcwd(), check=True)
                subprocess.run([GENFCC_PATH, "-i", state2_fchk], cwd=os.getcwd(), check=True)
                subprocess.run([GENDIP_PATH, "-i", state2_fchk], cwd=os.getcwd(), check=True)
                # Look for newly made fcc input files
                state1_file = find_unique_file("*_gs.fcc")
                state2_file = find_unique_file("*_ex.fcc")
                eldip_file = find_unique_file("eldip*fchk")
                magdip_file = find_unique_file("magdip*fchk")
                if not state1_file or not state2_file or not eldip_file: # still nothing
                    return

    # Template for 'Mode0' folder (normal OPA, no pre-excitation)
    fcc_template_mode0 = """$$$
PROPERTY     =   OPA  ; OPA/EMI/ECD/CPL/RR/TPA/TPCD/MCD/IC/NRSC
MODEL        =   AH   ; AS/ASF/AH/VG/VGF/VH
DIPOLE       =   FC   ; FC/HTi/HTf
TEMP         =   0.00 ; (temperature in K) 
;DE           = (read) ; (adiabatic/vertical energy in eV. By default, read from state files) 
BROADFUN     =   GAU  ; GAU/LOR/VOI
HWHM         =   0.036 ; (broadening width in eV)
METHOD       =   TI   ; TI/TD
;VIBRATIONAL ANALYSIS 
NORMALMODES  =   COMPUTE   ; COMPUTE/READ/IMPLICIT
COORDS       =   CARTESIAN ; CARTESIAN/INTERNAL
;INPUT DATA FILES 
STATE1_FILE  =   ../../{state1_file}
STATE2_FILE  =   ../../{state2_file}
ELDIP_FILE   =   ../../{eldip_file}
"""

    # Template for Mode1 to ModeN folders (with pre-excitation)
    fcc_template_preexc = """$$$
PROPERTY     =   OPA  ; OPA/EMI/ECD/CPL/RR/TPA/TPCD/MCD/IC/NRSC
MODEL        =   AH   ; AS/ASF/AH/VG/VGF/VH
DIPOLE       =   FC   ; FC/HTi/HTf
TEMP         =   0.00 ; (temperature in K) 
;DE           = (read) ; (adiabatic/vertical energy in eV. By default, read from state files) 
BROADFUN     =   GAU  ; GAU/LOR/VOI
HWHM         =   0.036 ; (broadening width in eV)
METHOD       =   TI   ; TI/TD
;VIBRATIONAL ANALYSIS 
NORMALMODES  =   COMPUTE   ; COMPUTE/READ/IMPLICIT
COORDS       =   CARTESIAN ; CARTESIAN/INTERNAL
;INPUT DATA FILES 
STATE1_FILE  =   ../../{state1_file}
STATE2_FILE  =   ../../{state2_file}
ELDIP_FILE   =   ../../{eldip_file}
NQMODE_EXC   =   1
NMODE_EXC    =   {mode_number}
"""
    
    # Template for EMI (state1 and state2 files are flipped, no pre-excitation)
    fcc_template_emi = """$$$
PROPERTY     =   EMI  ; OPA/EMI/ECD/CPL/RR/TPA/TPCD/MCD/IC/NRSC
MODEL        =   AH   ; AS/ASF/AH/VG/VGF/VH
DIPOLE       =   FC   ; FC/HTi/HTf
TEMP         =   0.00 ; (temperature in K) 
;DE           = (read) ; (adiabatic/vertical energy in eV. By default, read from state files) 
BROADFUN     =   GAU  ; GAU/LOR/VOI
HWHM         =   0.036 ; (broadening width in eV)
METHOD       =   TI   ; TI/TD
;VIBRATIONAL ANALYSIS 
NORMALMODES  =   COMPUTE   ; COMPUTE/READ/IMPLICIT
COORDS       =   CARTESIAN ; CARTESIAN/INTERNAL
;INPUT DATA FILES 
STATE1_FILE  =   ../{state2_file}
STATE2_FILE  =   ../{state1_file}
ELDIP_FILE   =   ../{eldip_file}
FORCE_REAL   =   YES
"""

    # Template for MCD - assuming fixed geometries (e.g., from crystal structures)
    fcc_template_MCD = """$$$
PROPERTY     =   MCD  ; OPA/EMI/ECD/CPL/RR/TPA/TPCD/MCD/IC/NRSC
MODEL        =   VH   ; AS/ASF/AH/VG/VGF/VH
DIPOLE       =   {dipole}   ; FC/HTi/HTf
TEMP         =   0.00 ; (temperature in K) 
;DE           = (read) ; (adiabatic/vertical energy in eV. By default, read from state files) 
BROADFUN     =   GAU  ; GAU/LOR/VOI
HWHM         =   0.02 ; (broadening width in eV)
METHOD       =   TI   ; TI/TD
;VIBRATIONAL ANALYSIS 
NORMALMODES  =   COMPUTE   ; COMPUTE/READ/IMPLICIT
COORDS       =   CARTESIAN ; CARTESIAN/INTERNAL
;INPUT DATA FILES 
STATE1_FILE  =   ../../{state1_file}
STATE2_FILE  =   ../{state2_file}
ELDIP_FILE   =   ../{eldip_file}
MAGDIP_FILE  =   ../{magdip_file}
FORCE_REAL   =   YES
"""

    # Template for resonance Raman (for epr-SRS)
    fcc_template_RR = """$$$
PROPERTY     =   RR  ; OPA/EMI/ECD/CPL/RR/TPA/TPCD/MCD/IC/NRSC
MODEL        =   AH   ; AS/ASF/AH/VG/VGF/VH
DIPOLE       =   FC   ; FC/HTi/HTf
TEMP         =   0.00 ; (temperature in K) 
;DE           = (read) ; (adiabatic/vertical energy in eV. By default, read from state files) 
BROADFUN     =   GAU  ; GAU/LOR/VOI
HWHM         =   0.036 ; (broadening width in eV)
METHOD       =   TI   ; TI/TD
;VIBRATIONAL ANALYSIS 
NORMALMODES  =   COMPUTE   ; COMPUTE/READ/IMPLICIT
COORDS       =   CARTESIAN ; CARTESIAN/INTERNAL
;INPUT DATA FILES 
STATE1_FILE  =   {state1_file}
STATE2_FILE  =   {state2_file}
ELDIP_FILE   =   {eldip_file}
RR_NFIELD    =   1 ; 9 ; -0.5 through +3.5 eV in 0.5 eV steps 
"""

    # Make new folder for calculation type and move there
    os.makedirs(calctype, exist_ok=True)
    os.chdir(calctype)
    
    ########## One-photon absorption/BonFIRE ##########
    if calctype == "OPA":
        # Create Mode0 folder first (no NMODE_EXC, NQMODE_EXC lines)
        folder_name_mode0 = "Mode0"
        os.makedirs(folder_name_mode0, exist_ok=True)
        
        # Check if previously run
        if os.path.isfile(f"{folder_name_mode0}/Assignments.dat"):
            print('FCclasses OPA assignments found in Mode0 - delete old folder to rerun.')
        else:
            # Prepare contents for fcc.inp file
            fcc_content_mode0 = fcc_template_mode0.format(state1_file=state1_file, 
                                                           state2_file=state2_file, 
                                                           eldip_file=eldip_file)
            file_path_mode0 = os.path.join(folder_name_mode0, "fcc.inp")
        
            # Write fcc.inp file for Mode0
            with open(file_path_mode0, "w") as file:
                file.write(fcc_content_mode0)
        
            print(f"Created: {folder_name_mode0}/fcc.inp")
        
            # Run "fcclasses3 fcc.inp" inside Mode0
            try:
                subprocess.run([FC_CLASSES_PATH, "fcc.inp"], cwd=folder_name_mode0, check=True)
                print(f"Successfully ran 'fcclasses3 fcc.inp' in {folder_name_mode0}")
            except subprocess.CalledProcessError as e:
                print(f"Error running 'fcclasses3 fcc.inp' in {folder_name_mode0}: {e}")
            except FileNotFoundError:
                print(f"Error: The specified path '{FC_CLASSES_PATH}' for 'fcclasses3' was not found.")
    
        # Create Mode1 to ModeN folders
        for i in range(1, n + 1):
            folder_name = f"Mode{i}"
            os.makedirs(folder_name, exist_ok=True)
            
            # Check if previously run
            if os.path.isfile(f"Mode{i}/Assignments.dat"):
                print(f"FCclasses OPA assignments found in Mode{i} - delete old folder to rerun.")
            else:
                fcc_content = fcc_template_preexc.format(state1_file=state1_file, 
                                                             state2_file=state2_file, 
                                                             eldip_file=eldip_file, 
                                                             mode_number=i)
                file_path = os.path.join(folder_name, "fcc.inp")
        
                with open(file_path, "w") as file:
                    file.write(fcc_content)
        
                print(f"Created: {folder_name}/fcc.inp")
        
                # Run "fcclasses3 fcc.inp" inside the folder
                try:
                    subprocess.run([FC_CLASSES_PATH, "fcc.inp"], cwd=folder_name, check=True)
                    print(f"Successfully ran 'fcclasses3 fcc.inp' in {folder_name}")
                except subprocess.CalledProcessError as e:
                    print(f"Error running 'fcclasses3 fcc.inp' in {folder_name}: {e}")
                except FileNotFoundError:
                    print(f"Error: The specified path '{FC_CLASSES_PATH}' for 'fcclasses3' was not found.")
 
    ########## Magnetic circular dichroism ##########
    elif calctype == "MCD":
        
        # Set calculation type
        folder_name = "FC"
        # Create folder
        os.makedirs(folder_name, exist_ok=True)
        # Prepare contents for fcc.inp files
        fcc_content = fcc_template_MCD.format(state1_file=state1_file, 
                                                     state2_file=state2_file, 
                                                     eldip_file=eldip_file, 
                                                     magdip_file=magdip_file,
                                                     dipole=folder_name)
        # Write fcc.inp
        file_path = os.path.join(folder_name, "fcc.inp")
        with open(file_path, "w") as file:
            file.write(fcc_content)
        print(f"Created: {folder_name}/fcc.inp")
        # Run calculation
        try:
            subprocess.run([FC_CLASSES_PATH, "fcc.inp"], cwd=folder_name, check=True)
            print(f"Successfully ran 'fcclasses3 fcc.inp' in {folder_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error running 'fcclasses3 fcc.inp' in {folder_name}: {e}")
        except FileNotFoundError:
            print(f"Error: The specified path '{FC_CLASSES_PATH}' for 'fcclasses3' was not found.")
            
        # Set calculation type
        folder_name = "HTi"
        # Create folder
        os.makedirs(folder_name, exist_ok=True)
        # Prepare contents for fcc.inp files
        fcc_content = fcc_template_MCD.format(state1_file=state1_file, 
                                                     state2_file=state2_file, 
                                                     eldip_file=eldip_file, 
                                                     magdip_file=magdip_file,
                                                     dipole=folder_name)
        # Write fcc.inp
        file_path = os.path.join(folder_name, "fcc.inp")
        with open(file_path, "w") as file:
            file.write(fcc_content)
        print(f"Created: {folder_name}/fcc.inp")
        # Run calculation
        try:
            subprocess.run([FC_CLASSES_PATH, "fcc.inp"], cwd=folder_name, check=True)
            print(f"Successfully ran 'fcclasses3 fcc.inp' in {folder_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error running 'fcclasses3 fcc.inp' in {folder_name}: {e}")
        except FileNotFoundError:
            print(f"Error: The specified path '{FC_CLASSES_PATH}' for 'fcclasses3' was not found.")
                
        # Set calculation type
        folder_name = "HTf"
        # Create folder
        os.makedirs(folder_name, exist_ok=True)
        # Prepare contents for fcc.inp files
        fcc_content = fcc_template_MCD.format(state1_file=state1_file, 
                                                     state2_file=state2_file, 
                                                     eldip_file=eldip_file, 
                                                     magdip_file=magdip_file,
                                                     dipole=folder_name)
        # Write fcc.inp
        file_path = os.path.join(folder_name, "fcc.inp")
        with open(file_path, "w") as file:
            file.write(fcc_content)
        print(f"Created: {folder_name}/fcc.inp")
        # Run calculation
        try:
            subprocess.run([FC_CLASSES_PATH, "fcc.inp"], cwd=folder_name, check=True)
            print(f"Successfully ran 'fcclasses3 fcc.inp' in {folder_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error running 'fcclasses3 fcc.inp' in {folder_name}: {e}")
        except FileNotFoundError:
            print(f"Error: The specified path '{FC_CLASSES_PATH}' for 'fcclasses3' was not found.")
            
    ########## Emission ##########
    elif calctype == "EMI":
        # Prepare contents for fcc.inp files
        fcc_content = fcc_template_emi.format(state1_file=state1_file, 
                                                     state2_file=state2_file, 
                                                     eldip_file=eldip_file)
        # Write fcc.inp to current folder
        file_path = os.path.join("fcc.inp")
        with open(file_path, "w") as file:
            file.write(fcc_content)
        print("Created: fcc.inp")
        # Run calculation
        try:
            subprocess.run([FC_CLASSES_PATH, "fcc.inp"], cwd=os.getcwd(), check=True)
            print("Successfully ran 'fcclasses3 fcc.inp' for fluorescence emission.")
        except subprocess.CalledProcessError as e:
            print(f"Error running 'fcclasses3 fcc.inp' for fluorescence emission: {e}")
        except FileNotFoundError:
            print(f"Error: The specified path '{FC_CLASSES_PATH}' for 'fcclasses3' was not found.")

    ########## Resonance Raman ##########
    elif calctype == "RR":
        # Prepare contents for fcc.inp files
        fcc_content = fcc_template_RR.format(state1_file=state1_file, 
                                                     state2_file=state2_file, 
                                                     eldip_file=eldip_file)
        # Write fcc.inp to current folder
        file_path = os.path.join("fcc.inp")
        with open(file_path, "w") as file:
            file.write(fcc_content)
        print("Created: fcc.inp")
        # Run calculation
        try:
            subprocess.run([FC_CLASSES_PATH, "fcc.inp"], cwd=os.getcwd(), check=True)
            print("Successfully ran 'fcclasses3 fcc.inp' for resonance Raman.")
        except subprocess.CalledProcessError as e:
            print(f"Error running 'fcclasses3 fcc.inp' for resonance Raman: {e}")
        except FileNotFoundError:
            print(f"Error: The specified path '{FC_CLASSES_PATH}' for 'fcclasses3' was not found.")

########## Processing/spectrum generation ##########

def create_contour_plot(base_folder, num_modes, plotting="yes"):
    """Reads data from spec_Int_TI.dat files in subfolders and creates a contour plot."""

    mode_numbers = range(num_modes + 1)
    energies = []
    intensities = []
    mode_indices = []
    
    # Loop through each folder and check for "spec_Int_TI.dat"
    for mode in mode_numbers:  # Includes Mode0 to ModeN
        folder_name = f"{base_folder}/Mode{mode}"
        file_path = os.path.join(folder_name, "spec_Int_TI.dat")
        
        if os.path.exists(file_path):  # If the file exists in the folder
            # Step 1: Read the data file using sep='\s+'
            df = pd.read_csv(file_path, sep='\s+', header=None, encoding="utf-8-sig")
            
            # Step 2: Assign column names
            df.columns = ["Energy (eV)", "Intensity"]
            
            # Step 3: Collect energy, intensity, and mode number for contour plot
            energies.extend(df["Energy (eV)"])
            intensities.extend(df["Intensity"])
            mode_indices.extend([mode] * len(df))

    if plotting == "yes":
        # Create the contour plot
        plt.figure(figsize=(6, 5))
        plt.tricontourf(mode_indices, energies, intensities, cmap="viridis")
        plt.colorbar(label="Intensity")
        
        plt.xlabel("Mode Number", fontsize=8, fontname="Arial")
        plt.ylabel("Energy (eV)", fontsize=8, fontname="Arial")
        plt.xticks(fontsize=6, fontname="Arial")
        plt.yticks(fontsize=6, fontname="Arial")
        
        plt.tick_params(direction="in", length=3, width=0.8)  # Inward ticks
        plt.tick_params(which="minor", length=1.5, width=0.5)  # Subtle minor ticks
        plt.minorticks_on()  # Enable minor ticks
        plt.grid(False)  # No major grid lines
    
        # Remove unnecessary borders (top and right spines)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_linewidth(0.8)
        plt.gca().spines["bottom"].set_linewidth(0.8)
    
        # Show the plot
        plt.show()
    
    return energies, intensities, mode_indices

def process_assignments_file(file_path): # used in FC_extractor
    """Extracts data from individual Assignments.dat files."""
    # Read the text file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Prepare a list to store the extracted data
    data = []

    # Regular expression to match the main data lines
    pattern = r"^\s*(\d+\.\d+)\s+(\d+\.\d+e[+-]?\d+|\d+\.\d+)\s+(\d+\.\d+e[+-]?\d+|\d+\.\d+)\s+(\d+\.\d+e[+-]?\d+|\d+\.\d+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.e\+]+)"

    # Regular expressions to match the Osc2 and Nqu2 values (extract numbers after the '=' sign)
    osc2_pattern = r"Osc2=\s*(\d+)"
    nqu2_pattern = r"Nqu2=\s*(\d+)"

    # Parse each line and extract the main data and Osc2/Nqu2 information
    osc2_values = []
    nqu2_values = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            # Extract the relevant data (Index, EN2, EN1, DE, En-0-0, FC, Spectrum)
            row = match.groups()
            data.append(row)

        # Extract Osc2 values
        osc2_matches = re.findall(osc2_pattern, line)
        if osc2_matches:
            osc2_values.append(osc2_matches)  # Store each match

        # Extract Nqu2 values
        nqu2_matches = re.findall(nqu2_pattern, line)
        if nqu2_matches:
            nqu2_values.append(nqu2_matches)  # Store each match

    # Insert a row of zeros for the first set of Osc2 and Nqu2 values
    osc2_values.insert(0, ['0'] * 7)  # 7 zeros for the 7 Osc2 columns
    nqu2_values.insert(0, ['0'] * 7)  # 7 zeros for the 7 Nqu2 columns

    # Create a DataFrame with appropriate column names for main data
    columns = ['INDEX', 'EN2(eV)', 'EN1(eV)', 'DE(eV)', 'En-0-0(cm-1)', 'FC', 'SPECTRUM']
    df = pd.DataFrame(data, columns=columns)

    # Reorder the columns to have 'En-0-0(cm-1)' and 'FC' first
    df = df[['En-0-0(cm-1)', 'FC', 'INDEX', 'EN2(eV)', 'EN1(eV)', 'DE(eV)', 'SPECTRUM']]

    # Convert numeric columns to appropriate types (float)
    numeric_columns = ['En-0-0(cm-1)', 'FC', 'INDEX', 'EN2(eV)', 'EN1(eV)', 'DE(eV)', 'SPECTRUM']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Create DataFrames for Osc2 and Nqu2 and make sure they have the same number of rows as the main DataFrame
    osc2_df = pd.DataFrame(osc2_values, columns=[f'Osc2_{i+1}' for i in range(7)])
    nqu2_df = pd.DataFrame(nqu2_values, columns=[f'Nqu2_{i+1}' for i in range(7)])

    # Convert Osc2 and Nqu2 columns to float
    osc2_df = osc2_df.apply(pd.to_numeric, errors='coerce')
    nqu2_df = nqu2_df.apply(pd.to_numeric, errors='coerce')

    # Interleave the Osc2 and Nqu2 columns
    interleaved_columns = []
    for i in range(7):
        interleaved_columns.append(f'Osc2_{i+1}')
        interleaved_columns.append(f'Nqu2_{i+1}')

    # Create a new DataFrame by interleaving the columns
    interleaved_df = pd.concat([osc2_df, nqu2_df], axis=1)
    interleaved_df = interleaved_df[interleaved_columns]  # Reorder the columns as interleaved

    # Concatenate the interleaved Osc2/Nqu2 DataFrame with the main DataFrame
    df = pd.concat([df, interleaved_df], axis=1)

    return df

def FC_extractor():
    """Find all "Assignments.dat" files in the current directory and subdirectories"""
    assignments_files = glob.glob(os.path.join(os.getcwd(), '**', 'Assignments.dat'), recursive=True)
    
    # Create an Excel writer to write all the data into a single Excel file
    with pd.ExcelWriter('Extracted_FCfactors.xlsx') as writer:
        for assignments_file in assignments_files:
            # Process each Assignments.dat file using separate function
            df = process_assignments_file(assignments_file)
            
            # Use the folder name as the Excel sheet name
            folder_name = os.path.basename(os.path.dirname(assignments_file))
            
            # Write each DataFrame to a separate sheet
            df.to_excel(writer, sheet_name=folder_name, index=False)
            df.to_csv('Extracted_FCfactors.csv', index=False)
    
    #print("All Assignment.dat files processed and saved into 'Extracted_FCfactors.xlsx'.")

def spec_combiner():
    """Combines spectra from spec_Int_TI.dat files into a single .csv."""
    # Get the current working directory as the parent directory
    parent_directory = os.getcwd()
    
    # Initialize an empty list to hold the data
    energy_list = intensity_list = []
    combined_spectra = pd.DataFrame()
    
    # Loop through each folder and check for "spec_Int_TI.dat"
    for folder in os.listdir(parent_directory):
        folder_path = os.path.join(parent_directory, folder)
        
        # Check for the spec_Int_TI.dat files
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, "spec_Int_TI.dat")
            
            if os.path.exists(file_path):  # If the file exists in the folder
                # Read the data file using sep='\s+'
                df = pd.read_csv(file_path, sep='\s+', header=None, encoding="utf-8-sig")
                
                # Assign column names
                df.columns = ["Energy (eV)", "Intensity"]
                
                # Add the folder's Energy and Intensity to the combined DataFrame
                folder_name = os.path.basename(folder_path)  # Get the name of the parent folder
                # combined_spectra[f"{folder_name} (eV)"] = df["Energy (eV)"]
                # combined_spectra[f"{folder_name} Intensity"] = df["Intensity"]
                
                # More efficient collection from data frame into a list
                df["Energy (eV)"].rename(f"{folder_name} Energy (eV)", inplace=True)
                energy_list.append(df["Energy (eV)"])
                df["Intensity"].rename(f"{folder_name} Intensity", inplace=True)
                intensity_list.append(df["Intensity"])
    
    # Concatenate into data frame
    energy_df = pd.concat(intensity_list, axis=1)
    intensity_df = pd.concat(intensity_list, axis=1)
    combined_spectra = pd.concat([combined_spectra, energy_df, intensity_df], axis=1)
    
    # Save the combined DataFrame to a .csv file
    combined_spectra.to_csv("combined_spectra.csv", index=False)
    
    #print("Spectra have been successfully combined and saved as 'combined_spectra.csv'.")
    
def parse_gaussian_log(filename):
    """
    Parses Gaussian log files according to:
    Frequencies, Red. masses, Frc consts, IR Inten, Raman Activ, Depolar (P), Depolar (U)
    Filters out imaginary frequencies.
    """
    frequencies = []
    ir_intensities = []
    raman_activities = []


    with open(filename, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if 'Frequencies --' in line:
            freqs = [float(x) for x in line.split()[2:]]

            # Look ahead for IR Inten and Raman Activ
            ir_vals = []
            raman_vals = []
            for j in range(i+1, min(i+10, len(lines))):
                l = lines[j]
                if 'IR Inten' in l:
                    ir_vals = [float(x) for x in l.split()[3:]]
                if 'Raman Activ' in l:
                    raman_vals = [float(x) for x in l.split()[3:]]
                    break

            # Ensure lengths match
            if len(freqs) == len(ir_vals) == len(raman_vals):
                for f, ir, r in zip(freqs, ir_vals, raman_vals):
                    if f > 0:  # remove imaginary
                        frequencies.append(f)
                        ir_intensities.append(ir)
                        raman_activities.append(r)
            elif len(freqs) == len(ir_vals):
                for f, ir in zip(freqs, ir_vals):
                    if f > 0:  # remove imaginary
                        frequencies.append(f)
                        ir_intensities.append(ir)
                        raman_activities.append(0)
            else:
                print(f"Warning: unequal lengths in {filename} at line {i}")

    return np.array(frequencies), np.array(ir_intensities), np.array(raman_activities)

def find_log_files(folder):
    log_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.log'):
                log_files.append(os.path.join(root, file))
    return log_files

def create_combined_csv(folder):
    """
    Creates a CSV with all modes from all log files.
    Each log file gets 3 columns: Mode frequency, IR intensity, Raman activity
    """
    log_files = find_log_files(folder)
    if not log_files:
        print("No .log files found.")
        return

    all_data = {}
    max_len = 0

    for log in log_files:
        freqs, ir_int, raman_act = parse_gaussian_log(log)
        max_len = max(max_len, len(freqs))
        base_name = os.path.splitext(os.path.basename(log))[0]

        all_data[f"{base_name} Mode frequency (cm^-1)"] = list(freqs)
        all_data[f"{base_name} IR intensity (km/mol)"] = list(ir_int)
        all_data[f"{base_name} Raman activity (AU)"] = list(raman_act)

    # Pad shorter lists
    for key in all_data:
        if len(all_data[key]) < max_len:
            all_data[key] += [np.nan]*(max_len - len(all_data[key]))

    df = pd.DataFrame(all_data)
    df.to_csv("combined_modes.csv", index=False)
    print("Saved combined_modes.csv with all vibrational modes.")

def gaussian_broadening(freqs, intensities, x, fwhm=10):
    """
    Returns Gaussian-broadened spectrum
    """
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    spectrum = np.zeros_like(x)
    for f,inten in zip(freqs,intensities):
        spectrum += inten*np.exp(-(x-f)**2/(2*sigma**2))
    return spectrum

def plot_spectra(folder, freq_range=(2000,2300), fwhm=10, mode='both', scale_factor=1.0):
    """
    Plots spectra for all log files with optional frequency scaling.
    - mode: 'IR', 'Raman', or 'both'
    - scale_factor: multiply all frequencies for plotting (empirical correction)
    Annotations are added at the right side of the figure, color-matched to each spectrum.
    """
    log_files = find_log_files(folder)
    if not log_files:
        print("No .log files found.")
        return

    x = np.linspace(freq_range[0], freq_range[1], 1000)
    plt.figure(figsize=(3, 6))

    if mode in ['IR','both']:
        offset = 0
        if mode == 'both':
            ax_ir = plt.subplot(2,1,1)
        else:
            ax_ir = plt.subplot(1,1,1)
        for log in log_files:
            freqs, ir_int, _ = parse_gaussian_log(log)
            freqs_plot = freqs * scale_factor
            mask = (freqs_plot >= freq_range[0]) & (freqs_plot <= freq_range[1])
            freqs_plot, ir_int_plot = freqs_plot[mask], ir_int[mask]
            spectrum = gaussian_broadening(freqs_plot, ir_int_plot, x, fwhm)
            if spectrum.max() > 0:
                spectrum /= spectrum.max()
            line, = ax_ir.plot(x, spectrum + offset, color=None)
            # Add annotation on the right side
            ax_ir.text(freq_range[1]*0.995, offset + 0.02, os.path.basename(log),
                       verticalalignment='bottom', horizontalalignment='right',
                       color=line.get_color(), fontsize=8)
            offset += 1.05
        ax_ir.set_xlabel(f"Frequency (cm$^{{-1}}$) × {scale_factor}")
        ax_ir.set_ylabel("Normalized IR Intensity + offset")
        ax_ir.set_title(f"IR spectra ({freq_range[0]}-{freq_range[1]} cm^-1, scaled)")

    if mode in ['Raman','both']:
        offset = 0
        if mode == 'both':
            ax_raman = plt.subplot(2,1,2)
        else:
            ax_raman = plt.subplot(1,1,1)
        for log in log_files:
            freqs, _, raman_act = parse_gaussian_log(log)
            freqs_plot = freqs * scale_factor
            mask = (freqs_plot >= freq_range[0]) & (freqs_plot <= freq_range[1])
            freqs_plot, raman_plot = freqs_plot[mask], raman_act[mask]
            spectrum = gaussian_broadening(freqs_plot, raman_plot, x, fwhm)
            if spectrum.max() > 0:
                spectrum /= spectrum.max()
            line, = ax_raman.plot(x, spectrum + offset, color=None)
            labelstring = os.path.basename(log).split('_linked.log')[0]
            labelstring = labelstring.split('.log')[0]
            ax_raman.text(freq_range[1]*0.995, offset + 0.02, labelstring,
                          verticalalignment='bottom', horizontalalignment='right',
                          color=line.get_color(), fontsize=8)
            offset += 1.05
        # ax_raman.set_xlabel(f"Frequency (cm$^{{-1}}$) × {scale_factor}")
        ax_raman.set_xlabel("Wavenumber (cm$^{{-1}}$)")
        # ax_raman.set_ylabel("Normalized Raman Intensity + offset")
        ax_raman.set_ylabel("Normalized Raman Intensity (a.u.)")
        #ax_raman.set_title(f"Raman spectra ({freq_range[0]}-{freq_range[1]} cm^-1, scaled)")
    
    plt.tight_layout()
    plt.show()    
    
################## Main ##################

if __name__ == "__main__":
    
    ########## Input parsing ##########
    
    # Called as: python (this_script).py --cores $(nproc) --mem $(nmem) --deuterate F --jobtype gsonly --FCtype OPA --preexcite yes --Nmodes 1 --wmin 2000 --wmax 2400 --scaling 0.953 --rerunFC yes --anharmonic resumeCN --bonfire yes
    parser = argparse.ArgumentParser(description="Determine available computing power.")
    parser.add_argument('--cores', type=int, help='Number of available processing cores.')
    parser.add_argument('--mem', type=float, help='Available memory in GB.')
    parser.add_argument('--deuterate', type=str, help='Target atom to replace with deuteriums.')
    parser.add_argument('--jobtype', type=str, help='DFT job type to run.')
    parser.add_argument('--FCtype', type=str, help='FCclasses job type to run.')
    parser.add_argument('--preexcite', type=str, help='Whether to run calculations with vibrational pre-excitations.')
    parser.add_argument('--Nmodes', type=int, help='Specify number of modes to do vibrational pre-excitations.')
    parser.add_argument('--wmin', type=float, help='Minimum frequency cutoff (for mode of interest).')
    parser.add_argument('--wmax', type=float, help='Maximum frequency cutoff (for mode of interest).')
    parser.add_argument('--scaling', type=float, help='Vibrational frequency scaling factor.')
    parser.add_argument('--rerunFC', type=str, help='Whether to force rerunning FCclasses calculations.')
    parser.add_argument('--anharmonic', type=str, help='Whether to run anharmonic calculations, and if resuming from a previous calculation or not.')
    parser.add_argument('--bonfire', type=str, help='Whether to generate a BonFIRE spectrum or not.')
    args = parser.parse_args()

    if args.cores:
        cores = min([31,args.cores]) # no more than 31 cores
    if args.mem:
        ram = min([190,math.floor(args.mem)]) # no more than 190 GB RAM
    if args.deuterate:
        deuterate = args.deuterate # atom to replace with deuterium
        deuterate = deuterate.ljust(5)
    if args.jobtype:
        jobtype = args.jobtype
    else:
        jobtype = None
    if args.FCtype:
        FCtype = args.FCtype
    else:
        FCtype = "EMI" # default to emission
    if args.preexcite:
        preexcite = args.preexcite
        if args.Nmodes: # if pre-exciting and modes specified
            Nmodes = args.Nmodes
        else:
            Nmodes = "all" # run all modes if unspecified
    else:
        preexcite = "no" # default to no pre-excitation
        Nmodes = 0 # no pre-excitations
    if args.wmin:
        wmin = args.wmin
    else:
        wmin = 2000
    if args.wmax:
        wmax = args.wmax
    else:
        wmax = 2400
    if args.scaling:
        scaling = args.scaling
    else:
        scaling = 0.953 # for triple-bond; 0.97 for double-bond
    if args.rerunFC:
        rerunFC = args.rerunFC
    else:
        rerunFC = "no" # default to using previous
    if args.anharmonic:
        if args.anharmonic == "resumeCN":
            anharmonic = args.anharmonic
        else:
            anharmonic = "fromscratch"
    else:
        anharmonic = "no"
    if args.bonfire:
        bonfire = args.bonfire
    else:
        bonfire = "no" # default to no bonfire

    ########## DFT ##########
    print('####### Beginning DFT calculations. #######')
    # Get subfolders of current folder
    working_dir = os.getcwd()
    print("Working directory:", str(working_dir))
    
    #for folder, dirs, files in os.walk(working_dir):
    for folder in os.listdir(working_dir): # loop through folders
    
        # Ignore hidden folders (those starting with '.')
        if os.path.isdir(folder) and not folder.startswith('.'):
            os.chdir(folder)
            print("Current folder: ", str(folder))
            
            # Look for previous Gaussian calculations
            found_chks = look_for_files("chk")
            
            if found_chks:
                print("Found Gaussian checkpoint files – assuming DFT calculations have already completed.", str(found_chks))
            else: # run DFT calculations
                # Look for ChemDraws
                found_cdxs = look_for_files("cdx")
                
                # Make sure ChemDraws were found
                if found_cdxs:
                    # Loop over .cdxs in the folder
                    for current_file in found_cdxs: # file.cdx
                        file_base = os.path.splitext(current_file)[0]  # remove only the .cdx extension
                        
                        ###### Run jobs ######
                        print("##### Beginning OpenBabel conversions #####")
                        # Convert to SMILES
                        smiles_file = f"{file_base}.smi"
                        run_cmd([OBABEL_PATH, str(current_file), "-O", str(smiles_file)])
                        
                        # Fix negative charge parsing errors (read as +255 instead of -1)
                        with open(smiles_file, 'r', encoding='utf-8') as f:
                            smiles = f.read()
                            
                        cleaned_smiles = smiles.replace("+255", "-")
                        
                        with open(smiles_file, 'w', encoding='utf-8') as f:
                            f.write(cleaned_smiles)
                        
                        # Read total charge via oreport - run on cleaned SMILES output
                        run_cmd([OBABEL_PATH, str(smiles_file), "-oreport", "-O", f"{file_base}_oreport.txt"])
                        with open(f"{file_base}_oreport.txt",'r',errors='ignore') as file:
                            lines = file.readlines()
                        for i, line in enumerate(lines):
                            if i == 4:
                                charge_line = str(line)
                                split_charge = charge_line.split(": ")
                                if split_charge[0] == 'TOTAL CHARGE':
                                    mol_charge = int(split_charge[1])
                                else:
                                    mol_charge = 0
                                    
                        # Build 2D geometry
                        mol2d_file = f"{file_base}_2d.mol"
                        run_cmd([OBABEL_PATH, str(smiles_file), "-O", str(mol2d_file), "--gen2d"])
                        
                        # Build 3D geometry
                        mol3d_file = f"{file_base}.mol"
                        run_cmd([OBABEL_PATH, str(mol2d_file), "-O", str(mol3d_file), "--gen3d"])
                        
                        # Coarse optimization
                        opt_file = f"{file_base}_opt.mol"
                        run_cmd([OBABEL_PATH, str(mol3d_file), "-O", str(opt_file), "--minimize"])
                        
                        # Make Gaussian job file
                        gjf_path = make_gjf(opt_file, str(file_base), mol_charge, mol_spin, cores, ram, jobtype)
    
                        # Replace with deuteriums
                        if args.deuterate:
                            replace_with_deuteriums(gjf_path, deuterate)
                            print(f"Replaced {deuterate} with deuteriums in {gjf_path}")
    
                        # Run Gaussian job
                        if oncluster == 1:
                            print("##### Beginning Gaussian calculations #####")
                            run_cmd([G16_PATH, str(gjf_path)])
                            print(f"Gaussian calculation on {file_base} completed successfully.")
                        
                else: # no .cdx files found
                    found_cdxmls = look_for_files("cdxml",folder)
                    if found_cdxmls:
                        print("ERROR: Please convert to .cdx files and try again.")
                    else:
                        print(f"ERROR: No ChemDraw files found in {folder}.")
            
            # Return to working directory for next loop iteration
            os.chdir(working_dir)

    ########## FCclasses ##########
    print('####### Beginning FCclasses processing. #######')
    
    # Loop through folders and run FCclasses
    for folder in os.listdir(working_dir):
        if os.path.isdir(folder) and not folder.startswith('.'):
            os.chdir(folder)
            print("Current folder: ", str(folder))
            
            # Decide number of modes
            logfile = find_unique_file("*_link*.log")      
            print(logfile)
            freqs, ir_int, raman_act = parse_gaussian_log(logfile) # reads gs and ex freqs
            if Nmodes == "all":
                N = int(len(freqs)/2)
            else: # ensure requested modes not more than normal modes present
                N = min([len(freqs)/2,Nmodes])
            
            # Check if FCC already run - FCtype will be OPA, EMI, RR, or MCD
            if os.path.isfile(f"{FCtype}/Assignments.dat") and rerunFC == "no":
                print('FCclasses assignments already present – moving on to next folder.')
            else: 
                if FCtype == "RR":
                    batch_run_FCC(N,"RR") # run RR first for preRR_Int file
                    os.chdir(folder)
                    batch_run_FCC(N,"EMI") # run EMI after RR to get FCFs
                else:
                    batch_run_FCC(N,f"{FCtype}")
            
            FC_extractor() # extracts FC factors from Assignments.dat files
            if FCtype == "OPA" and preexcite == "yes":
                create_contour_plot(".", N) # visualize data in a contour plot
                spec_combiner() # combines spectra from spec_Int_TI.dat (or TD) files
            
            # Return to working directory for next loop iteration
            os.chdir(working_dir)
    
    ########## Mode analysis ##########
    print('####### Completed FCclasses processing – now collecting FC factors. #######')

    # Extract FC factors but start from IR frequencies
    newdf2 = pd.DataFrame()
    
    for i, folder in enumerate(os.listdir(working_dir)): # loop through folders
        if os.path.isdir(folder) and not folder.startswith('.'):
            os.chdir(folder)
            print("Current folder: ", str(folder))
            newdf2.loc[i,'Folder'] = str(folder)
            
            guess_output = 'EMI/fcc.out'
            
            if os.path.isfile(guess_output):
                # Load output file
                lines = bf.gen_utils.readlines(guess_output)
                # with open(guess_output, 'r') as f:
                #     lines = f.readlines()
                
                # Look for vibrational frequencies
                freq_start1 = freq_end1 = freq_start2 = freq_end2 = None
                for j, line in enumerate(lines):
                    if not freq_start1 and 'FREQUENCIES (cm-1)' in line:
                        freq_start1 = j + 2
                    if freq_start1 and not freq_end1 and 'THERMOCHEMISTRY' in line:
                        freq_end1 = j - 4
                    if not freq_start2 and freq_end1 and 'FREQUENCIES (cm-1)' in line:
                        freq_start2 = j + 2
                    if freq_start2 and not freq_end2 and 'MINIMINZING DIFFERENCES BETWEEN STATES CARTESIAN GEOMS' in line:
                        freq_end2 = j - 10
                        
                mode_number = None
                # Look through frequencies
                for m in range(freq_start1, freq_end1 + 1):
                    if not mode_number:
                        freqline = lines[m].strip()
                        temp = freqline.split()
                        try:
                            frequency1 = scaling*float(temp[1])
                            mode = int(temp[0])
                        except (ValueError, IndexError):
                            continue # skip lines are comments, etc.
                        if frequency1 > wmin and frequency1 < wmax:
                            mode_number = mode
                            # print(f"Found mode: {frequency} cm–1 (scaled) for {folder}.")
                if not mode_number:
                    print("No modes found within specified frequency range - please check bounds.")
                else:
                    # Mode found - continue with normal processing
                    newdf2.loc[i,'ModeNumber'] = mode_number
                    newdf2.loc[i,'FrequencyState1'] = frequency1
                    
                    # Find frequency in State2
                    frequency2 = None
                    for m in range(freq_start2, freq_end2 + 1):
                        if not frequency2:
                            freqline = lines[m].strip()
                            temp = freqline.split()
                            try:
                                frequency = scaling*float(temp[1])
                                mode = int(temp[0])
                            except (ValueError, IndexError):
                                continue # skip lines are comments, etc.
                            if mode == mode_number:
                                frequency2 = frequency
                                # print(f"Found mode: {frequency} cm–1 (scaled) for {folder}.")
                    
                    newdf2.loc[i,'FrequencyState2'] = frequency2
                    
                    if frequency1 and frequency2:
                        frequency = max(frequency1,frequency2)
                        newdf2.loc[i,'Frequency'] = frequency
                    
                    # FC factors
                    guess_csv = 'EMI/Extracted_FCfactors.csv'
                    
                    if os.path.isfile(guess_csv):
                        # Read csv
                        df = pd.read_csv(guess_csv)
                        
                        # Loop through CSV
                        for index, row in df.iterrows():
                            
                            # Look for mode in specified energy range
                            mode_energy = scaling*abs(row['En-0-0(cm-1)'])
                            
                            # Look only for desired mode
                            if row['Osc2_1'] == mode_number and row['Nqu2_1'] == 1 and row['Nqu2_2'] == 0:
                                newdf2.loc[i,'FCfactor'] = row['FC']
                            
                            # Get 0-0 FC
                            if mode_energy == 0:
                                newdf2.loc[i,'FC00'] = row['FC']

                    
                    # HR factors and displacements
                    guess_HR = 'EMI/HuangRhys.dat'
                    if os.path.isfile(guess_HR):
                        # Read HR file as whitespace-delimited data frame
                        df2 = pd.read_csv(guess_HR, sep='\s+')
                        df3 = df2.rename(columns={'#':'ModeNumber', 'Mode':'HRfactor'})
                        
                        # Loop through data frame
                        for index, row in df3.iterrows():
                            if mode_number == row['ModeNumber']:
                                HRfactor = row['HRfactor']
                                # if HRfactor < 0:
                                #     print(HRfactor)
                                #     print("Problem folder: ", str(folder))
                                displacement = np.sqrt(HRfactor*2)
                                if mode_number == newdf2.loc[i,'ModeNumber']:
                                    newdf2.loc[i,'HuangRhys'] = HRfactor
                                    newdf2.loc[i,'Displacement'] = displacement

                    # Pre-resonance Raman
                    guess_preRR = 'RR/preRR_Int_ShortTimeDynamics.dat'
                    if os.path.isfile(guess_preRR):
                        # Read preRR data file
                        with open(guess_preRR, 'r') as f:
                            lines = f.readlines()
                        
                        for j, line in enumerate(lines):
                            dataline = lines[j].strip()
                            temp = dataline.split()
                            try:
                                frequency = scaling*float(temp[0])
                                intensity = float(temp[1])
                                if round(frequency) == round(newdf2.loc[i,'Frequency']):
                                    newdf2.loc[i,'PreRR_Int'] = intensity
                            except (ValueError, IndexError):
                                continue
            
            # Return to working directory for next loop iteration
            os.chdir(working_dir)

    fig, ax = bf.plot.bfplot()
    for i, row in newdf2.iterrows():
        if abs(row['FCfactor']**2) > 0 and abs(row['FC00']**2) > 0:
            FCR = abs(row['FCfactor']**2)/abs(row['FC00']**2)
            ax.plot(row['Frequency'], FCR, marker='o', markersize=8, linewidth=2)
            newdf2.loc[i,'FCratio'] = FCR
    ax.set_xlabel_custom('Frequency scaled by '+str(scaling)+' (cm$^{–1}$)')
    ax.set_ylabel_custom('FCR')
    plt.subplots_adjust(left=0.2, right=0.9, top=0.88, bottom=0.19, wspace=0.2, hspace=0.2)
    plt.show()
    
    newdf2.to_csv("Collected_FCfactors.csv", index=False)
    
    print(f"##### Finished extracting FC factors of modes between {wmin} and {wmax} cm–1. #####")
    
    ########## Anharmonic (resume from previous) ##########
    if anharmonic == "resumeCN":
        print("##### Beginning anharmonic calculations (resuming from previous calculations). #####")
        
        anharm_template = """%nprocshared={cores}
%mem={ram}GB
%oldchk={file_base}_gs.chk
%chk={file_base}_anharmCN.chk
# opt freq=(noraman,savenormalmodes,anharmonic,selectanharmonicmodes) b3lyp/6-31g(d,p) scrf=(smd,solvent=dmso) nosymm pop=nbo geom=allcheck



{file_base} nitrile anharmonic

modes={mode_number}


"""
        for i, folder in enumerate(os.listdir(working_dir)): # loop through folders
            if os.path.isdir(folder) and not folder.startswith('.'):
                os.chdir(folder)
                guess_gschk = f"{folder}_gs.chk"
                guess_log = f"{folder}_linked.log"
                if os.path.isfile(guess_gschk) and os.path.isfile(guess_log):
                    # Parse log and find nitrile
                    freqs = ir_ints = ramans = []
                    freqs, ir_ints, ramans = parse_gaussian_log(guess_log)
                    for i in range(0,len(freqs)/2):
                        if freqs[i] > wmin and freqs[i] < wmax:
                            nitrile_mode = i+1
                    
                    # Calculate anharmonic mode number (3N-6 - mode + 1)
                    mode_number = len(freqs)-nitrile_mode+1
                    
                    # Make job file
                    anharm_content = anharm_template.format(cores=cores, ram=ram, file_base=folder, mode_number=mode_number)
                    anharm_job_path = f"{folder}_anharmCN.gjf"
                    
                    with open(anharm_job_path, "w") as file:
                        file.write(anharm_content)
                
                    print(f"Created: {anharm_job_path}")
                    
                    # Run Gaussian job
                    if oncluster == 1:
                        print("##### Beginning Gaussian calculations #####")
                        run_cmd([G16_PATH, str(gjf_path)])
                        print(f"Anharmonic Gaussian calculation on {folder} completed.")
                    else:
                        print(f"Not running on cluster - anharmonic Gaussian calculation on {folder} not completed.")
                    
                else:
                    print(f"##### Couldn't find previous .chk or .log to resume for {folder} - proceeding to next folder. #####")
                
                # Return to working directory for next loop iteration
                os.chdir(working_dir)

    ########## BonFIRE spectrum generation ##########
    if bonfire == "yes":
        print("##### Beginning BonFIRE spectrum generation. #####")
        for i, folder in enumerate(os.listdir(working_dir)): # loop through folders
            if os.path.isdir(folder) and not folder.startswith('.'):
                os.chdir(folder)
                # Load IR spectrum from Gaussian log
                guess_log = f"{folder}_linked.log"
                freqs = ir_ints = ramans = []
                freqs, ir_ints, ramans = parse_gaussian_log(guess_log)
                # Loads gs and ex - keep gs only
                max_modes = int(len(freqs)/2)
                freqs = freqs[0:max_modes]
                ir_ints = ir_ints[0:max_modes]
                
                # Plot FTIR
                fig, ax = bf.plot.bfplot()
                ax.plot(freqs, ir_ints, marker='o', markersize=8, linewidth=2)
                ax.set_xlabel_custom('Frequency (cm$^{–1}$)')
                ax.set_ylabel_custom('Intensity')
                plt.subplots_adjust(left=0.2, right=0.9, top=0.88, bottom=0.19, wspace=0.2, hspace=0.2)
                plt.show()
                
                if not os.path.isdir("OPA"):
                    print("##### Couldn't find OPA results for {folder} - proceeding to next folder. #####")
                else:
                    os.chdir("OPA")
                    energies, intensities, mode_indices = create_contour_plot(".",max_modes,"no")
                    
                    all_spectra = pd.DataFrame()
                    all_spectra['ModeNumber'] = mode_indices
                    all_spectra['Energy (eV)'] = energies
                    all_spectra['Frequency (cm-1)'] = all_spectra['Energy (eV)']*8065.56 # eV to cm-1
                    all_spectra['Intensity'] = intensities
                    
                    # Filter dataframe into UV-vis and pre-excited OPA
                    uv_vis = all_spectra[all_spectra['ModeNumber'] == 0]
                    opa_spectra = all_spectra[all_spectra['ModeNumber'] > 0]
                    
                    # Plot UV-vis
                    fig, ax = bf.plot.bfplot()
                    ax.plot(uv_vis['Frequency (cm-1)'], uv_vis['Intensity'], marker='o', markersize=8, linewidth=2)
                    ax.set_xlabel_custom('Frequency (cm$^{–1}$)')
                    ax.set_ylabel_custom('Intensity')
                    plt.subplots_adjust(left=0.2, right=0.9, top=0.88, bottom=0.19, wspace=0.2, hspace=0.2)
                    plt.show()
                    
                    # Crop to pre-resonance regime
                    uvv_peak = uv_vis.loc[uv_vis['Intensity'].idxmax()]
                    preresonance_cutoff = uvv_peak['Frequency (cm-1)'] - 1000
                    cropped_opa = opa_spectra[opa_spectra['Frequency (cm-1)'] < preresonance_cutoff]
                    
                    # Confirm crop with contour
                    plt.figure(figsize=(6, 5))
                    plt.tricontourf(cropped_opa['ModeNumber'], cropped_opa['Frequency (cm-1)'], cropped_opa['Intensity'], cmap="viridis")
                    plt.colorbar(label="Intensity")
                    plt.xlabel("Mode Number", fontsize=8, fontname="Arial")
                    plt.ylabel("Frequency (cm$^{–1}$)", fontsize=8, fontname="Arial")
                    plt.xticks(fontsize=6, fontname="Arial")
                    plt.yticks(fontsize=6, fontname="Arial")
                    plt.show()
                    
                    # Isolate up-conversion peaks
                    bfdf = pd.DataFrame()
                    bfdf['IR frequencies (cm-1)'] = freqs
                    bfdf['IR intensities (km/mol)'] = ir_ints
                    probe_peaks = []
                    for i in range(0,len(freqs)):
                        crop_peaks = cropped_opa[cropped_opa['ModeNumber'] == i+1].max()
                        bfdf.loc[i, 'Up-conversion intensities'] = crop_peaks['Intensity']
                    
                    # Apply IR frequencies and intensities
                    bfdf['BonFIRE'] = bfdf['IR intensities (km/mol)']*bfdf['Up-conversion intensities']
                    
                    # Plot BonFIRE
                    fig, ax = bf.plot.bfplot()
                    ax.plot(bfdf['IR frequencies (cm-1)'], bfdf['BonFIRE']/bfdf['BonFIRE'] .max(), linewidth=2)
                    ax.set_xlabel_custom('IR frequency (cm$^{–1}$)')
                    ax.set_ylabel_custom('BonFIRE (AU)')
                    plt.subplots_adjust(left=0.2, right=0.9, top=0.88, bottom=0.19, wspace=0.2, hspace=0.2)
                    plt.show()
                    
                    # Apply Gaussian broadening
                    bf_spec = pd.DataFrame()
                    bf_spec['Frequency (cm-1)'] = np.linspace(0,3500,7000)
                    bf_spec['BonFIRE'] = gaussian_broadening(bfdf['IR frequencies (cm-1)'], bfdf['BonFIRE'], bf_spec['Frequency (cm-1)'])
                    
                    # Plot BonFIRE
                    fig, ax = bf.plot.bfplot()
                    ax.plot(bf_spec['Frequency (cm-1)']*scaling, bf_spec['BonFIRE']/bf_spec['BonFIRE'].max(), linewidth=2)
                    ax.set_xlabel_custom('Frequency scaled by '+str(scaling)+' (cm$^{–1}$)')
                    ax.set_ylabel_custom('BonFIRE (AU)')
                    plt.subplots_adjust(left=0.2, right=0.9, top=0.88, bottom=0.19, wspace=0.2, hspace=0.2)
                    plt.show()
                    
                    # Save data frames
                    bfdf.to_csv('BonFIRE_stick_spectra.csv', index=False)
                    bf_spec.to_csv('BonFIRE_broadened_spectra.csv', index=False)
                    print("##### Successfully generated and exported BonFIRE spectra for {folder}. #####")
                    
                # Return to working directory for next loop iteration
                os.chdir(working_dir)
        
        
