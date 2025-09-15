#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:39:58 2025

@author: pkocheril

Fully automated batch DFT for FCclasses.

Call as: python (this_script).py --cores $(nproc) --mem $(nmem) --deuterate F --jobtype gsonly

Optional arguments:
--cores $(nproc) # number of processor cores
--mem $(nmem) # available memory in GB
--deuterate F # atom to replace with deuterium
--jobtype (gsonly, raman, default) # only do ground-state, Raman, or default (gs+ex for FCclasses)
"""

import os
import subprocess
from pathlib import Path
from collections import defaultdict
import glob
import argparse
import math

################## Setup ##################

mol_charge = 0 # 0 by default, will update if needed
mol_spin = 1 # singlets by default
cores = 4 # 4 by default, will use argparse to get nproc
ram = 4 # 4 by default, will use argparse to get nmem (custom script)

solvent = "dmso" # implicit solvent to be used
solvent_model = "smd" # definitely use SMD for FCclasses

# Set paths
if os.path.isdir("/resnick/groups/WeiLab/software/"): # if on the cluster
    OBABEL_PATH = "/resnick/groups/WeiLab/software/openbabel/bin/obabel"
    G16_PATH = "/resnick/groups/WeiLab/software/g16/g16"
    oncluster = 1 # 1 = true
else:
    OBABEL_PATH = "/usr/local/openbabel/bin/obabel"
    G16_PATH = "/Applications/GaussView6/gv/g16"
    oncluster = 0


################## Functions ##################

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

    # Define all available job types with descriptive names
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
    }

    # Define job type configurations
    job_configs = {
        "gsonly": ["rough_opt", "ground_state"],
        "raman": ["rough_opt", "raman"],
        "default": ["rough_opt", "ground_state", "excited_state"],
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

################## Main ##################

if __name__ == "__main__":
    
    ########## Input parsing ##########
    
    # Called as: python (this_script).py --cores $(nproc) --mem $(nmem) --deuterate target_atom --jobtype (gsonly, raman, default)
    parser = argparse.ArgumentParser(description="Determine available computing power.")
    parser.add_argument('--cores', type=int, help='Number of available processing cores.')
    parser.add_argument('--mem', type=float, help='Available memory in GB.')
    parser.add_argument('--deuterate', type=str, help='Target atom to replace with deuteriums.')
    parser.add_argument('--jobtype', type=str, help='Job type to run.')
    args = parser.parse_args()

    if args.cores:
        cores = min([31,args.cores])
    if args.mem:
        ram = min([190,math.floor(args.mem)])
    if args.deuterate:
        deuterate = args.deuterate # atom to replace with deuterium
        deuterate = deuterate.ljust(5)
    if args.jobtype:
        jobtype = args.jobtype
    else:
        jobtype = None

    ########## DFT ##########
    
    # Get subfolders of current folder
    working_dir = os.getcwd()
    print("Working directory:", str(working_dir))
    
    #for folder, dirs, files in os.walk(working_dir):
    for folder in os.listdir(working_dir): # loop through folders
    
    #print("Current directory", str(folder))
    
        # Ignore hidden folders (those starting with '.')
        if os.path.isdir(folder) and not folder.startswith('.'):
            os.chdir(folder)
            print("Current folder: ", str(folder))
            
            # Look for ChemDraws
            found_cdxs = look_for_files("cdx")
            
            # Make sure ChemDraws were found
            if found_cdxs:
                #print("Found ChemDraws", str(found_cdxs))
                
                # Loop over .cdxs in the folder
                for current_file in found_cdxs: # file.cdx
                    file_base = os.path.splitext(current_file)[0]  # remove only the .cdx extension
                    
                    ###### Run jobs ######
                    print("##### Beginning OpenBabel conversions #####")
                    # Convert to SMILES
                    smiles_file = f"{file_base}.smi"
                    run_cmd([OBABEL_PATH, str(current_file), "-O", str(smiles_file)])
                    
                    # Build 2D geometry
                    mol2d_file = f"{file_base}_2d.mol"
                    run_cmd([OBABEL_PATH, str(smiles_file), "-O", str(mol2d_file), "--gen2d"])
                    
                    # Build 3D geometry
                    mol3d_file = f"{file_base}.mol"
                    run_cmd([OBABEL_PATH, str(mol2d_file), "-O", str(mol3d_file), "--gen3d"])
                    
                    # Coarse optimization
                    opt_file = f"{file_base}_opt.mol"
                    run_cmd([OBABEL_PATH, str(mol3d_file), "-O", str(opt_file), "--minimize"])
                    
                    # Read total charge via oreport
                    run_cmd([OBABEL_PATH, str(current_file), "-oreport", "-O", f"{file_base}_oreport.txt"])
                    with open(f"{file_base}_oreport.txt",'r',errors='ignore') as file:
                        lines = file.readlines()
                    for i, line in enumerate(lines):
                        if i == 4:
                            charge_line = str(line)
                            split_charge = charge_line.split(": ")
                            if split_charge[0] == 'TOTAL CHARGE':
                                mol_charge = int(split_charge[1])
                                if mol_charge > 4: # from weird parsing errors
                                    mol_charge = 0
                            else:
                                mol_charge = 0
                    
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
            
            
            # Return to working directory
            os.chdir(working_dir)
    

