#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Indrek Kalvet
ikalvet@uw.edu
"""
import pyrosetta as pyr
import pyrosetta.rosetta
import numpy as np
import os
import sys
import time
import queue
import threading
import argparse


def getSASA(pdbfile, resno=None, pose=True):
    """
    Takes in a PDB file and calculates its SASA.

    Procedure by Brian Coventry
    """
    if not pose:
        pose = pyr.pose_from_file(pdbfile)
    if pose:
        pose = pdbfile

    atoms = pyr.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    n_ligands = 0
    for res in pose.residues:
        if res.is_ligand():
            n_ligands += 1

    for i, res in enumerate(pose.residues):
        if resno is None:
            atoms.resize(i+1, res.natoms(), True)
        else:
            # Only considering the asked residue, protein chains and
            # if the asked residue is not ligand, then the first ligand state,
            # if available.
            chain_no = res.chain()
            if i+1 == resno and res.is_ligand():
                atoms.resize(i+1, res.natoms(), True)
            elif res.is_ligand() and n_ligands > 1 and i+1 != resno:
                atoms.resize(i+1, res.natoms(), False)
            elif res.is_ligand() and n_ligands == 1:
                atoms.resize(i+1, res.natoms(), True)
            elif res.is_protein() and len(pose.chain_sequence(chain_no)) > 1:
                atoms.resize(i+1, res.natoms(), True)
            elif res.is_protein() and len(pose.chain_sequence(chain_no)) == 1 and i+1 != resno:
                atoms.resize(i+1, res.natoms(), False)
            elif res.is_protein() and len(pose.chain_sequence(chain_no)) == 1 and i+1 == resno:
                atoms.resize(i+1, res.natoms(), True)
            else:
                print(i+1)

    surf_vol = pyr.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)

    if resno is not None:
        res_surf = 0
        for i in range(1, pose.residue(resno).natoms()+1):
            res_surf += surf_vol.surf(resno, i)
        return res_surf
    else:
        return surf_vol


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pdb", nargs="+", type=str, required=True, help="Input PDBs from Matcher")
    parser.add_argument("--params", nargs="+", type=str, help="Params files of ligands and noncanonicals")
    parser.add_argument("--flags", type=str, help="Matcher flags file used for generating these input matches")
    parser.add_argument("--limit", default=0.35, type=float, help="Cutoff for relative SASA filtering")
    parser.add_argument("--nproc", default=os.cpu_count(), type=int, help="# of CPU cores used")
    parser.add_argument("--analyze", default=False, action="store_true", help="Perform only analysis and do not move PDB files?")
    
    args = parser.parse_args()
    
    pdbfiles = args.pdb
    params = args.params
    NPROC = args.nproc
    SASA_limit = args.limit
    only_analyze = args.analyze

    if args.flags is not None:
        flagfile = open(args.flags, 'r').readlines()
        if args.params is None:
            params = []
            for l in flagfile:
                if '.params' in l:
                    params.append(l.split()[1])

    extra_res_fa = ""
    if len(params) > 0:
        extra_res_fa = "-extra_res_fa"
        for p in params:
            extra_res_fa += " " + p

    bad_match_dir = "bad_matches"
    if os.path.exists('bad_matches'):
        if only_analyze is False:
            bad_match_dir = "bad_matches_2"


    print(extra_res_fa)

    DAB = None
    DAB = "/home/ikalvet/Rosetta/DAlphaBall.gcc"
    
    assert DAB is not None, "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n"\
                            "For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"

    pyr.init(f"{extra_res_fa} -mute all -dalphaball {DAB}")

    SASAs = {}

    start = time.time()

    ligand_found = False
    for p in params:
        par = open(p, 'r').readlines()
        for l in par:
            if "TYPE LIGAND" in l:
                ligand_pdb = p.replace(".params", ".pdb")
                ligand = pyr.pose_from_file(ligand_pdb)
                ligand_SASA = getSASA(ligand).tot_surf
                ligand_found = True
                break
        if ligand_found:
            break

    if not ligand_found:
        sys.exit("Was not able to find a ligand from the params files specified.")

    q = queue.Queue()

    for pdbfile in pdbfiles:
        q.put(pdbfile)

    def analyze_all_state_SASA():
        while True:
            pdbfile = q.get(block=True)
            if pdbfile is None:
                return
            SASAs[pdbfile] = {}
            pose = pyr.pose_from_file(pdbfile)

            chains = []
            for n in range(pose.num_chains()):
                chains.append(pose.chain_sequence(n+1))

            ligand_SASAs = []

            # Iterating over all ligand states in a pose and calculating
            # the ligand SASA.
            for i, res in enumerate(pose.residues):
                if res.is_ligand():
                    ligand_SASAs.append(getSASA(pose, i+1))

            SASAs[pdbfile]['all'] = ligand_SASAs
            SASAs[pdbfile]['all_rel'] = [x/ligand_SASA for x in ligand_SASAs]
            SASAs[pdbfile]['avg'] = np.average(ligand_SASAs)
            SASAs[pdbfile]['avg_rel'] = np.average(SASAs[pdbfile]['all_rel'])
            SASAs[pdbfile]['med_rel'] = np.median(SASAs[pdbfile]['all_rel'])

            # if the median of all ligand states relative SASA is smaller than SASA_limit,
            # then this match is considered good
            if SASAs[pdbfile]['med_rel'] < SASA_limit:
                SASAs[pdbfile]['pass'] = True
                print("{}: {:.3f} GOOD".format(pdbfile, SASAs[pdbfile]['med_rel']))
            else:
                SASAs[pdbfile]['pass'] = False
                print("{}: {:.3f} BAD".format(pdbfile, SASAs[pdbfile]['med_rel']))


    threads = [threading.Thread(target=analyze_all_state_SASA) for _i in range(NPROC)]

    for thread in threads:
        thread.start()
        q.put(None)  # one EOF marker for each thread

    for thread in threads:
        thread.join()

    end = time.time()

    print("Sorting the matches took {:.3f} seconds.".format(end - start))

    # Printing out a report file
    logfile = "match_analysis.log"
    if os.path.exists(logfile):
        logfile = "match_analysis2.log"

    with open(logfile, "w") as file:
        # Header line with column labels
        namelength = max([len(x) for x in SASAs.keys()]) + 2
        labels = ["Name", 'SASA', 'Pass']
        widths = [namelength, 7, 7]
        header = ""
        for i, (l, w) in enumerate(zip(labels, widths)):
            if i == 0:
                header += "{0:<{1}}".format(l, w)
            else:
                header += "{0:>{1}}".format(l, w)
        file.write(header+"\n")

        # Data for each PDB file
        good_count = 0
        for pdb in SASAs.keys():
            file.write("{0:<{1}}{2:>{3}.2f}"
                       "{4:>{5}}\n".format(pdb, namelength,
                                           SASAs[pdb]['avg_rel'], 7,
                                           str(SASAs[pdb]['pass']), 7))
            if SASAs[pdb]['pass'] is True:
                good_count += 1
    
    print(f"{good_count}/{len(SASAs)} matches passed the SASA <= {SASA_limit*100}% filter.")
    
    # If the script is running not only in analysis mode then bad matches are
    # moved to another directory.
    if not only_analyze:
        if not os.path.exists(bad_match_dir):
            os.mkdir(bad_match_dir)
        for pdb in SASAs.keys():
            if not SASAs[pdb]['pass']:
                os.rename(pdb, f'{bad_match_dir}/{pdb}')

if __name__ == "__main__":
    main()
