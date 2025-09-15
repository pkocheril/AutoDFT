#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch execution and post-processing of FCclasses3 
with vibrational pre-excitation

-- now bundled with AutoDFT

full_FCC_batch history:
v2 -- added MCD
v3 -- added EMI
v4 -- added batch EMI (and FC factor extraction for a target mode)
v5 -- added preresonance Raman and displacement factors
v6 -- added reruns
v7 -- fixed paths for HPC
"""

# Import modules
import os
import re
import glob
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


################## Setup ##################

# User variables
CALC_TYPE = 4 # 0 = OPA/pre-excitation, 1 = MCD, 2 = EMI (no modes), 
# 3 = batch EMI, 4 = batch RR
N = 0  # number of normal modes for OPA/pre-excitation
# Energy range (cm-1) for mode to extract (for batch EMI)
WMIN = 2000 # frequency cutoff (scaled)
WMAX = 2400 # frequency cutoff (scaled)
TARGET_MODE_COUNT = 1
SCALING_FACTOR = 0.953 # value to scale final frequencies by - 0.953 for triple bonds
RERUN = 0 # 0 = use existing, 1 = rerun analysis

# Set paths
if os.path.isdir("/resnick/groups/WeiLab/software/"): # if on the cluster
    OBABEL_PATH = "/resnick/groups/WeiLab/software/openbabel/bin/obabel"
    G16_PATH = "/resnick/groups/WeiLab/software/g16/g16"
    FC_CLASSES_PATH = "/resnick/groups/WeiLab/software/fcclasses3-3.0.3/src/main/fcclasses3"
    FORMCHK_PATH = "/resnick/groups/WeiLab/software/g16/formchk"
    GENFCC_PATH = "/resnick/groups/WeiLab/software/fcclasses3-3.0.3/FCCBIN/gen_fcc_state"
    GENDIP_PATH = "/resnick/groups/WeiLab/software/fcclasses3-3.0.3/FCCBIN/gen_fcc_dipfile"
    oncluster = 1 # 1 = true
else:
    OBABEL_PATH = "/usr/local/openbabel/bin/obabel"
    G16_PATH = "/Applications/GaussView6/gv/g16"
    FC_CLASSES_PATH = "/usr/local/bin/fcclasses3"
    FORMCHK_PATH = "/Applications/GaussView6/gv/formchk"
    GENFCC_PATH = "/usr/local/bin/gen_fcc_state"
    GENDIP_PATH = "/usr/local/bin/gen_fcc_dipfile"
    oncluster = 0


################## Functions ##################

# Define functions
def find_unique_file(pattern): # used in batch_run_FCC
    matches = glob.glob(pattern)
    if len(matches) == 0:
        #print(f"Error: No file found matching '{pattern}' in the current directory.")
        return None
    elif len(matches) > 1:
        #print(f"Error: Multiple files found matching '{pattern}': {matches}")
        return None
    return matches[0]

def batch_run_FCC(n,calctype):
    # Look for fcc input files in current folder
    state1_file = find_unique_file("*_gs*.fcc")  # gs file is STATE1_FILE
    state2_file = find_unique_file("*_ex*.fcc")  # ex file is STATE2_FILE
    eldip_file = find_unique_file("_eldip*fchk")
    magdip_file = find_unique_file("magdip*fchk")

    # Stop execution if any required file is missing or ambiguous
    if not state1_file or not state2_file or not eldip_file:
        # Look for .chk files in current folder
        state1_chk = find_unique_file("*_gs*.chk")
        state2_chk = find_unique_file("*_ex*.chk")
        
        if not state1_chk or not state2_chk:
            print("Unambiguous ground-state and excited-state files not found – exiting.")
            return
        else:
            # Generate fchk files
            subprocess.run([FORMCHK_PATH, state1_chk], cwd=os.getcwd(), check=True)
            subprocess.run([FORMCHK_PATH, state2_chk], cwd=os.getcwd(), check=True)
            # Look for fchk files
            state1_fchk = find_unique_file("*_gs*.fchk")
            state2_fchk = find_unique_file("*_ex*.fchk")
            if not state1_fchk or not state2_fchk:
                return
            else:
                # Generate state and dipole files
                subprocess.run([GENFCC_PATH, "-i", state1_fchk], cwd=os.getcwd(), check=True)
                subprocess.run([GENFCC_PATH, "-i", state2_fchk], cwd=os.getcwd(), check=True)
                subprocess.run([GENDIP_PATH, "-i", state2_fchk], cwd=os.getcwd(), check=True)
                # Look for newly made fcc input files
                state1_file = find_unique_file("*_gs*.fcc")
                state2_file = find_unique_file("*_ex*.fcc")
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
STATE1_FILE  =   ../{state1_file}
STATE2_FILE  =   ../{state2_file}
ELDIP_FILE   =   ../{eldip_file}
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
STATE1_FILE  =   ../{state1_file}
STATE2_FILE  =   ../{state2_file}
ELDIP_FILE   =   ../{eldip_file}
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
STATE1_FILE  =   {state2_file}
STATE2_FILE  =   {state1_file}
ELDIP_FILE   =   {eldip_file}
FORCE_REAL   =   YES
"""

    # Template for MCD
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
STATE1_FILE  =   ../{state1_file}
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

################## OPA with pre-excitation ##################
    if calctype == 0:
        # Create Mode0 folder first (no NMODE_EXC, NQMODE_EXC lines)
        folder_name_mode0 = "Mode0"
        os.makedirs(folder_name_mode0, exist_ok=True)
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
 
################## MCD ##################
    elif calctype == 1:
        
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
            
################## Fluorescence emission ##################
    elif calctype == 2:
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

################## Resonance Raman ##################     
    elif calctype == 4:
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

def create_contour_plot(base_folder, num_modes):
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


# def spec_combiner():
#     """Combines spectra from spec_Int_TI.dat files into a single .csv."""
#     # Get the current working directory as the parent directory
#     parent_directory = os.getcwd()
    
#     # Initialize an empty list to hold the data
#     combined_spectra = pd.DataFrame()  # This will be the final DataFrame with all columns
    
#     # Loop through each folder and check for "spec_Int_TI.dat"
#     for folder in os.listdir(parent_directory):
#         folder_path = os.path.join(parent_directory, folder)
        
#         # Check for the spec_Int_TI.dat files
#         if os.path.isdir(folder_path):
#             file_path = os.path.join(folder_path, "spec_Int_TI.dat")
            
#             if os.path.exists(file_path):  # If the file exists in the folder
#                 # Read the data file using sep='\s+'
#                 df = pd.read_csv(file_path, sep='\s+', header=None, encoding="utf-8-sig")
                
#                 # Assign column names
#                 df.columns = ["Energy (eV)", "Intensity"]
                
#                 # Add the folder's Energy and Intensity to the combined DataFrame
#                 folder_name = os.path.basename(folder_path)  # Get the name of the parent folder
#                 combined_spectra[f"{folder_name} (eV)"] = df["Energy (eV)"]
#                 combined_spectra[f"{folder_name} Intensity"] = df["Intensity"]
                
#     # Save the combined DataFrame to a .csv file
#     combined_spectra.to_csv("combined_spectra.csv", index=False)
    
#     #print("Spectra have been successfully combined and saved as 'combined_spectra.csv'.")



################## Main ##################
print('####### Beginning FCclasses processing. #######')
if CALC_TYPE > 2: # batch EMI or RR
    # Get subfolders of current folder
    working_dir = os.getcwd()
    print("Working directory:", str(working_dir))
    
    # Loop through folders and run FCclasses
    for folder in os.listdir(working_dir): # loop through folders
        if os.path.isdir(folder):
            os.chdir(folder)
            print("Current folder: ", str(folder))

            if os.path.isfile('Assignments.dat') and RERUN == 0:
                print('FCclasses assignments already present – moving on to next folder.')
            else:
                # Run FCC   
                if CALC_TYPE == 4:
                    batch_run_FCC(N,4) # run RR first for preRR_Int file
                batch_run_FCC(N,2) # then run EMI for FC factors
                FC_extractor()

            # Return to working directory for next loop iteration
            os.chdir(working_dir)
    
    print('####### Completed FCclasses processing – now collecting FC factors. #######')

    # Loop through folders and tabulate extracted FC factors
    newdf = pd.DataFrame()
    
    for i, folder in enumerate(os.listdir(working_dir)): # loop through folders
        if os.path.isdir(folder):
            os.chdir(folder)
            print("Current folder: ", str(folder))
            newdf.loc[i,'Folder'] = str(folder)
            
            guess_csv = 'Extracted_FCfactors.csv'
            
            if os.path.isfile(guess_csv):
                # Read each csv
                df = pd.read_csv(guess_csv)
                count = 0
                
                # Loop through CSV
                for index, row in df.iterrows():
                    
                    # Look for mode in specified energy range
                    mode_energy = SCALING_FACTOR*abs(row['En-0-0(cm-1)'])
                    if mode_energy > WMIN and mode_energy < WMAX and row['Nqu2_1'] == 1 and row['Nqu2_2'] == 0 and count < TARGET_MODE_COUNT:
                        #print(row)
                        newdf.loc[i,'ModeNumber'] = row['Osc2_1']
                        newdf.loc[i,'ModeEnergy'] = mode_energy
                        newdf.loc[i,'FCfactor'] = row['FC']
                        count = count + 1 # keep track of added modes
                
                if count < TARGET_MODE_COUNT: # if not enough modes found
                    print("Not enough modes found – looking through fcc.out frequencies.")
                    # First, try reading log file and looking for mode of interest from IR frequencies
                    if os.path.isfile("fcc.out"):
                        with open("fcc.out", 'r') as f:
                            lines = f.readlines()

                        # Locate frequencies
                        freq_start = freq_end = None
                        for j, line in enumerate(lines):
                            if not freq_start and 'FREQUENCIES (cm-1)' in line:
                                freq_start = j + 2
                            if freq_start and not freq_end and 'THERMOCHEMISTRY' in line:
                                freq_end = j - 4
                        
                        mode_number = None
                        # Look through frequencies
                        for m in range(freq_start, freq_end + 1):
                            if not mode_number:
                                freqline = lines[m].strip()
                                temp = freqline.split()
                                try:
                                    frequency = SCALING_FACTOR*float(temp[1])
                                    mode = int(temp[0])
                                except (ValueError, IndexError):
                                    continue # skip lines are comments, etc.
                                if frequency > WMIN and frequency < WMAX:
                                    mode_number = mode
                                    print(f"Found mode: {frequency} cm–1 (scaled) for {folder}.")
                        
                        if mode_number:
                            # Return to CSV for combination modes of mode_number
                            for index, row in df.iterrows():
                                
                                # Look for mode in specified energy range
                                mode_energy = SCALING_FACTOR*abs(row['En-0-0(cm-1)'])
                                
                                # Check for doubly excited combination modes first
                                if mode_energy > WMIN and mode_energy < WMAX and row['Osc2_1'] == mode_number and row['Nqu2_1'] == 1 and row['Nqu2_2'] == 1 and count < CALC_TYPE:
                                    #print(row)
                                    newdf.loc[i,'ModeNumber'] = row['Osc2_1']
                                    newdf.loc[i,'ModeEnergy'] = mode_energy
                                    newdf.loc[i,'FCfactor'] = row['FC']
                                    newdf.loc[i,'CombMode'] = row['Osc2_2']
                                    newdf.loc[i,'CombQuantum'] = row['Nqu2_2']
                                    count = count + 1 # keep track of added modes
                            
                            if count < TARGET_MODE_COUNT: # still not enough modes found
                                # Allow any arbitratily excited combination mode
                                for index, row in df.iterrows():
                                    
                                    # Look for mode in specified energy range
                                    mode_energy = SCALING_FACTOR*abs(row['En-0-0(cm-1)'])
                                    if mode_energy > WMIN and mode_energy < WMAX and row['Osc2_1'] == mode_number and row['Nqu2_1'] == 1 and count < CALC_TYPE:
                                        #print(row)
                                        newdf.loc[i,'ModeNumber'] = row['Osc2_1']
                                        newdf.loc[i,'ModeEnergy'] = mode_energy
                                        newdf.loc[i,'FCfactor'] = row['FC']
                                        newdf.loc[i,'CombMode'] = row['Osc2_2']
                                        newdf.loc[i,'CombQuantum'] = row['Nqu2_2']
                                        count = count + 1 # keep track of added modes
                            
                        else:
                            print(f"##### No modes found in fcc.out between {WMIN} and {WMAX} cm–1. #####")
                        
                    else:
                        # Loop through CSV again, allowing any combination modes
                        for index, row in df.iterrows():
                            
                            # Look for mode in specified energy range
                            mode_energy = SCALING_FACTOR*abs(row['En-0-0(cm-1)'])
                            if mode_energy > WMIN and mode_energy < WMAX and row['Nqu2_1'] == 1 and count < CALC_TYPE:
                                #print(row)
                                newdf.loc[i,'ModeNumber'] = row['Osc2_1']
                                newdf.loc[i,'ModeEnergy'] = mode_energy
                                newdf.loc[i,'FCfactor'] = row['FC']
                                count = count + 1 # keep track of added modes
            
            # HR -> displacement
            guess_HR = 'HuangRhys.dat'
            if os.path.isfile(guess_HR):
                # Read preRR data file
                with open(guess_HR, 'r') as f:
                    lines = f.readlines()
                
                for j, line in enumerate(lines):
                    dataline = lines[j].strip()
                    temp = dataline.split()
                    try:
                        modenum = int(temp[0])
                        HRfactor = float(temp[1])
                    except (ValueError, IndexError):
                        continue
                    displacement = np.sqrt(HRfactor*2)
                    if modenum == newdf.loc[i,'ModeNumber']:
                        newdf.loc[i,'Displacement'] = displacement
                        
            # Pre-resonance Raman
            guess_preRR = 'preRR_Int_ShortTimeDynamics.dat'
            if os.path.isfile(guess_preRR):
                # Read preRR data file
                with open(guess_preRR, 'r') as f:
                    lines = f.readlines()
                
                for j, line in enumerate(lines):
                    dataline = lines[j].strip()
                    temp = dataline.split()
                    try:
                        frequency = SCALING_FACTOR*float(temp[0])
                        intensity = float(temp[1])
                        if round(frequency) == round(newdf.loc[i,'ModeEnergy']):
                            newdf.loc[i,'PreRR_Int'] = intensity
                    except (ValueError, IndexError):
                        continue
            
            # Return to working directory for next loop iteration
            os.chdir(working_dir)
            
    print(f"##### Finished extracting FC factors of modes between {WMIN} and {WMAX} cm-1. #####")
    newdf.to_csv("Summary_FCfactors.csv", index=False)
    
    
    plt.plot(newdf['ModeEnergy'], abs(newdf['FCfactor']), 'r.')
    plt.xlabel('Mode energy scaled by '+str(SCALING_FACTOR)+' $(cm^{–1})$')
    plt.ylabel('|FC factor|')
    plt.show()
else: # OPA, MCD, or single EMI
    # Executing functions
    batch_run_FCC(N,CALC_TYPE) # run FCclasses
    
    if CALC_TYPE == 0: # for vibrational pre-excitations
        create_contour_plot(".", N) # visualize data in a contour plot
        FC_extractor() # extracts FC factors from Assignments.dat files
        spec_combiner() # combines spectra from spec_Int_TI.dat (or TD) files

    
    




