#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Indrek Kalvet
ikalvet@uw.edu
"""
import matplotlib.pyplot as plt
import sys
import os
import queue
import threading
import argparse
import numpy as np
import pandas as pd
from shutil import copy2
import pyrosetta as pyr
import pyrosetta.rosetta


comparisons = {'<=': '__le__',
               '<': '__lt__',
               '>': '__gt__',
               '>=': '__ge__',
               '=': '__eq__'}


def fileToList(filename, split=False):
    """
    Input: name of a file that has data in a list,
           must have newline after last line
    Output: clean list with file contents
    """
    l = open(filename, 'r').read()
    l = l.split('\n')
    l = l[:-1]
    if split is True:
        a = []
        for line in l:
            a.append(list(filter(None, line.split())))  
        if len(a[0]) == 1 and '.xyz' in filename:
            a = a[2:]
        return a
    return l


def parse_flags_file(flags_fname='', flags_list=[]):
    if flags_fname != '' and flags_list == []:
        flags_list = fileToList(flags_fname)
    elif flags_fname == '' and flags_list == []:
        print("Something went wrong with "
              "calling the parse_flags_file function.")
        sys.exit(1)
    flags = {}
    for l in flags_list:
        if '-extra_res_fa' in l:
            if 'params' not in flags.keys():
                flags['params'] = [l.split()[1]]
            else:
                flags['params'] = flags['params'] + [l.split()[1]]
        elif '-match::lig_name' in l:
            flags['ligand_name'] = l.split()[1]
        elif '-cstfile' in l:
            flags['cstfile'] = l.split()[1]
    return flags


def get_matcher_residues(filename):
    pdbfile = open(filename, 'r').readlines()

    matches = {}
    for l in pdbfile:
        if "ATOM" in l:
            break
        if "REMARK 666" in l:
            lspl = l.split()
            chain = lspl[9]
            res3 = lspl[10]
            resno = int(lspl[11])
            
            matches[resno] = {'name3': res3,
                              'chain': chain}
    return matches


def getResCoords_PyRosetta(pose, resno, names=False):
    resno = int(resno)
    coords = []
    crds = 0
    for l in str(pose.residue(resno)).split('\n'):
        if 'Atom Coordinates:' in l:
            crds = 1
        if crds == 1:
            coords.append(l)
    coords = coords[1:-2]

    xyz = []
    atomnames = []
    for l in coords:
        cspl = l.split(': ')
        atom = cspl[0].split()
        atomnames.append(atom[0])
        crd = cspl[1].split(', ')
        if atom[0][0].isnumeric():
            atom = atom[0][1]
        elif 'V' in atom[0]:
            continue
        else:
            atom = atom[0][:1]
        line = ([atom[0][:1], float(crd[0]), float(crd[1]), float(crd[2])])
        xyz.append(line)
    if names is True:
        return xyz, atomnames
    else:
        return xyz


def get_dihedral(a1, a2, a3, a4):
    """Praxeolitic formula from Stackexchange:
    https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python#
    1 sqrt, 1 cross product"""

    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)
    a4 = np.array(a4)

    b0 = a1 - a2
    b1 = a3 - a2
    b2 = a4 - a3

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return round(np.degrees(np.arctan2(y, x)), 1)


def get_angle(a1, a2, a3):
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)

    ba = a1 - a2
    bc = a3 - a2

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return round(np.degrees(angle), 1)


def getSASA(pose, ligand=False, atomlist=None):
    """
    Takes in a PDB file and calculates its SASA.

    If ligand mode is requested with 'ligand = 1'
    then only the SASA of the ligand inside the pocket is returned.

    Procedure by Brian Coventry
    """
    atoms = pyrosetta.rosetta.core.id.AtomID_Map_bool_t()
    atoms.resize(pose.size())

    for i in range(1, pose.size()+1):
        atoms.resize(i, pose.residue(i).natoms(), True)

    surf_vol = pyrosetta.rosetta.core.scoring.packing.get_surf_vol(pose, atoms, 1.4)

    if ligand is True:
        ligand_surf = 0.0
        if atomlist is None:
            for i in range(1, pose.residue(pose.size()).natoms()+1):
                ligand_surf += surf_vol.surf(pose.size(), i)
        else:
            for i in atomlist:
                ligand_surf += surf_vol.surf(pose.size(), i)
        return ligand_surf
    else:
        return surf_vol


def dump_scorefile(df, filename):
    widths = {}
    for k in df.keys():
        if k in ["SCORE:", "description", "name"]:
            widths[k] = 0
        if len(k) >= 12:
            widths[k] = len(k) + 1
        else:
            widths[k] = 12

    with open(filename, "w") as file:
        title = ""
        for k in df.keys():
            if k == "SCORE:":
                title += k
            elif k in ["description", "name"]:
                continue
            else:
                title += f"{k:>{widths[k]}}"
        if all([t not in df.keys() for t in ["description", "name"]]):
            title += f" {'description'}"
        elif "description" in df.keys():
            title += f" {'description'}"
        elif "name" in df.keys():
            title += f" {'name'}"
        file.write(title + "\n")
        
        for index, row in df.iterrows():
            line = ""
            for k in df.keys():
                if isinstance(row[k], (float, np.float16)):
                    val = f"{row[k]:.3f}"
                else:
                    val = row[k]
                if k == "SCORE:":
                    line += val
                elif k in ["description", "name"]:
                    continue
                else:
                    line += f"{val:>{widths[k]}}"
                
            if all([t not in df.keys() for t in ["description", "name"]]):
                line += f" {index}"
            elif "description" in df.keys():
                line += f" {row['description']}"
            elif "name" in df.keys():
                line += f" {row['name']}"
            file.write(line + "\n")


def get_score_df(scorefile):
    df = pd.read_csv(scorefile, sep='\s+', header=0)
    return df


def filter_scores(scores, filters):
    filtered_scores = scores.copy()

    for s in filters.keys():
        if filters[s] is not None and s in scores.keys():
            val = filters[s][0]
            sign = comparisons[filters[s][1]]
            filtered_scores =\
              filtered_scores.loc[(filtered_scores[s].__getattribute__(sign)(val))]
            n_passed = len(scores.loc[(scores[s].__getattribute__(sign)(val))])
            print(f"{s:<24} {filters[s][1]:<2} {val:>7.2f}: {len(filtered_scores)} "
                  f"designs left. {n_passed} pass ({(n_passed/len(scores))*100:.0f}%).")
    return filtered_scores


def main():    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pdb", nargs="+", type=str, help="Input PDB files")
    parser.add_argument("--scorefile", type=str, help="Design output scorefile")
    parser.add_argument("--params", nargs="+", type=str, help="Params files of ligands and noncanonicals")
    parser.add_argument("--flags", type=str, help="Enzdes flags file used for generating these input designs")
    parser.add_argument("--filters", type=str, help="File with user defined filters. If not defined then hardcoded Rosetta score filters will be used. Designs passing these filters will be scored for additional metrics and copied to a different location.")
    parser.add_argument("--nproc", default=os.cpu_count(), type=int, help="# of CPU cores used")


    N_THREADS = os.cpu_count()
    
    ### IMPORTANT VARIABLES ###
    
    args = parser.parse_args()
    
    pdbfiles = args.pdb
    params = args.params
    N_THREADS = args.nproc
    filterfile = args.filters

    ligand = sys.argv[sys.argv.index('--ligand')+1]

    WDIR = os.getcwd()

    ### IMPORTANT VARIABLES ###

    ### ROSETTA STUFF ###
    flaglist = fileToList(args.flags)

    flags = parse_flags_file(flags_list=flaglist)
    
    if 'params' in flags.keys():
        params = flags['params']

    extra_res_fa = '-extra_res_fa'
    for r in params:
        extra_res_fa += f' {r}'
        if ligand in r.split('/')[-1]:
            ligand_pdbfile = r.replace('.params', '.pdb')


    DAB = None
    DAB = "/home/ikalvet/Rosetta/DAlphaBall.gcc"
    
    assert DAB is not None, "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n"\
                            "For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"

    pyr.init(f"-dalphaball {DAB} {extra_res_fa} -mute all -beta_nov16")
    
    ### ROSETTA STUFF ###
    

    ### LOADING SCOREFILE ###
    
    scores_from_file = open(args.scorefile, 'r').readlines()
    scores_from_file = [x.split() for x in scores_from_file]
    
    scores = pd.DataFrame(data=scores_from_file[1:],
                          columns=scores_from_file[0],
                          dtype=float)
    
    ### LOADING SCOREFILE ###
    
    print(f"\n\n::::::: Analyzing {ligand} results :::::::\n")
    
    # Allowing user to define custom filtering cutoffs with a file
    if filterfile is not None:
        user_defined_filters = open(filterfile, 'r').readlines()
        user_defined_filters = [x.split() for x in user_defined_filters if len(x) > 1]
        interesting_scores = {}
        for line in user_defined_filters:
            if line[1] == "None":
                interesting_scores[line[0]] = None
            else:
                interesting_scores[line[0]] = [float(line[2]), line[1]]
    else:
        scores['score_per_res'] = np.nan
        if any([x in WDIR for x in ['onlyH', 'only_His']]):
            ### Scores for only HIS binding
            for index, row in scores.iterrows():
                scores.at[index, 'score_per_res'] = row.total_score / row.SR_2
    
            interesting_scores = {'all_cst': [3.0, '<='],
                                  'score_per_res': [-3.0, '<='],
                                  'nlr_SR1_rms': [0.6, '<='],
                                  'nlr_totrms': [0.8, '<=']}
    
        else:
            for index, row in scores.iterrows():
                scores.at[index, 'score_per_res'] = row.total_score / row.SR_2
            
            interesting_scores = {'all_cst': [3.0, '<='],
                                  'score_per_res': [-3.0, '<='],
                                  'nlr_SR1_rms': [0.6, '<='],
                                  'nlr_SR3_rms': [0.6, '<='],
                                  'nlr_totrms': [0.8, '<=']}

    ### FILTERING SCORES ###
    
    print(f"Number designs in the dataset: {len(scores)}")
    
    filtered_scores = filter_scores(scores, interesting_scores)

    
    if len(filtered_scores) == 0:
        filtered_scores =\
         scores.loc[(scores['all_cst'] <= interesting_scores['all_cst'][0])]
    
    ### FILTERING SCORES ###

    
    ligand_sasa = getSASA(pyr.pose_from_file(ligand_pdbfile)).tot_surf
    
    processed_scores = None
    if os.path.exists("all_scores.sc"):
        processed_scores = get_score_df("all_scores.sc")
    
    ### SETTING UP PARALLEL PROCESSING ###
    q = queue.Queue()
    
    for index, row in filtered_scores.iterrows():
        if processed_scores is not None:
            if row.description in processed_scores.description.values:
                continue
        q.put([index, row])
    
    print(f"Number of designs to be analyzed: {q.qsize()}\n")
    
    
    
    ### STARTING PARALLEL PROCESSING ###
    def analyze():
        while True:
            s = q.get(block=True)
            if s is None:
                return
    
            index = s[0]
            name = s[1]['description']
    
            print("Analyzing: {}".format(name))
    
            # Calculating SASA
            pdbfname = name + '.pdb'
    
            mutations = get_matcher_residues(pdbfname)
    
            pose = pyr.pose_from_file(pdbfname)
            ligand_surf = getSASA(pose, ligand=True)
            rel_lig_surf = ligand_surf/ligand_sasa
    
            scores.at[index, 'L_SASA'] = rel_lig_surf
            filtered_scores.at[index, 'L_SASA'] = rel_lig_surf
    
            lig_seqpos = pose.size()
            
            his_seqpos = None
            ED_seqpos = None
            for resno in mutations:
                if mutations[resno]['name3'] == "HIS":
                    his_seqpos = resno
                if mutations[resno]['name3'] in ["ASP", "GLU"]:
                    ED_seqpos = resno
    
            # Measuring Heme-coordinating HIS straightness
            if his_seqpos is not None:
                torsion_atoms = [(lig_seqpos, 'FE1'), (his_seqpos, 'NE2'),
                                  (his_seqpos, 'CE1'), (his_seqpos, 'ND1')]
                torsion_crds = [pose.residue(r).xyz(a) for r, a in torsion_atoms]
    
                his_torsion = abs(get_dihedral(*torsion_crds))
    
                scores.at[index, 'his_torsion'] = his_torsion
                filtered_scores.at[index, 'his_torsion'] = his_torsion
    
            ### Measuring HIS-E/D angle from the O that's closest to HN.
            # The N-H-O angle should be around 180 degrees
            if ED_seqpos is not None:
                ED_name1 = pose.residue(ED_seqpos).name1()
                angle_atoms = [[(his_seqpos, 'ND1'), (his_seqpos, 'HD1'), (ED_seqpos, f'O{ED_name1}1')],
                                [(his_seqpos, 'ND1'), (his_seqpos, 'HD1'), (ED_seqpos, f'O{ED_name1}2')]]
    
                OH_dists = {n: (pose.residue(his_seqpos).xyz('HD1') - pose.residue(ED_seqpos).xyz(f'O{ED_name1}{n}')).norm() for n in range(1, 3)}
                ED_atm = min(OH_dists, key=OH_dists.get)
    
                angle_crds = [pose.residue(r).xyz(a) for r, a in angle_atoms[ED_atm-1]]
    
                ED_angle = get_angle(*angle_crds)
    
                scores.at[index, 'ED_angle'] = ED_angle
                filtered_scores.at[index, 'ED_angle'] = ED_angle
    
            ### Finding H-bond donors around the COO groups of the Heme
            # Both COO groups must have a polar hydrogen within 2.5A of its oxygen
            HBond_res = {n: [] for n in range(1, 5)}

            for res in pose.residues:
                if res.seqpos() == pose.size():
                    break
                for n in range(1, 5):
                    an = f"O{n}"
                    if (pose.residue(lig_seqpos).xyz(an) - res.xyz('CA')).norm() < 10.0:
                        for polar_H in res.Hpos_polar_sc():
                            if (pose.residue(lig_seqpos).xyz(an) - res.xyz(polar_H)).norm() < 2.5:
                                HBond_res[n].append(res.seqpos())
                                break
            pairs = [(1, 2), (1, 4), (2, 3), (3, 4)]
            HB_test = any([(HBond_res[p[0]] != [] and HBond_res[p[1]] != []) for p in pairs])
    
            filtered_scores.at[index, 'COO_HB'] = float(HB_test)
            scores.at[index, 'COO_HB'] = float(HB_test)
    
            ### CALCULATING SHAPECOMP FOR FILTERED SCORES ###
            sc = pyrosetta.rosetta.protocols.simple_filters.ShapeComplementarityFilter()
            shapecomp = sc.compute(pose)
            filtered_scores.at[index, 'sc'] = shapecomp.sc
            scores.at[index, 'sc'] = shapecomp.sc
    
    
    if filterfile is None:
        scores['L_SASA'] = np.nan  # Relative SASA of the ligand
        # scores['ED_angle'] = np.nan  # Angle of the His-ED N-H..O interaction
        scores['COO_HB'] = np.nan
        scores['sc'] = np.nan
    
        threads = [threading.Thread(target=analyze) for _i in range(N_THREADS)]
        
        for thread in threads:
            thread.start()
            q.put(None)  # one EOF marker for each thread
        
        for thread in threads:
            thread.join()

        """
        Filtering based on custom metrics
        """
        interesting_scores['L_SASA'] = [0.20, '<=']
        interesting_scores['his_torsion'] = [150.0, '>=']
        interesting_scores['ED_angle'] = [150.0, '>=']
        interesting_scores['sc'] = [0.6, '>=']
        interesting_scores['COO_HB'] = [1, '=']
    
        filtered_scores2 = filtered_scores.copy()
    
        print("\nFiltering based on additional scores:")
        for s in interesting_scores.keys():
            if interesting_scores[s] is not None and s in filtered_scores.keys():
                val = interesting_scores[s][0]
                sign = comparisons[interesting_scores[s][1]]
                filtered_scores2 =\
                  filtered_scores2.loc[(filtered_scores2[s].__getattribute__(sign)(val))]
                print(f"{s:<18} {interesting_scores[s][1]:<2} {val:>7.2f}: {len(filtered_scores2)} designs left.")
    
        # Adding custom scores from the already processed DataFrame to the scores df.
        # This is done to make sure that the dumped scorefile contains all calculated scores.
        if processed_scores is not None:
            for index, row in scores.iterrows():
                if row.description in processed_scores.description.values:
                    processed_id = processed_scores.description.values.tolist().index(row.description)
                    if not np.isnan(processed_scores.iloc[processed_id]['L_SASA']):
                        for sc in interesting_scores.keys():
                            if sc in processed_scores.keys():
                                scores.at[index, sc] = processed_scores.at[processed_id, sc]
    else:
        filtered_scores2 = filtered_scores.copy()    

    dump_scorefile(scores, 'all_scores.sc')


    """
    Plotting scores used for filtering as histograms
    """
    scores2 = scores.loc[scores.all_cst <= 30.0]

    fig = plt.figure(figsize=(16, 16))
    items = len(interesting_scores)
    rows = items // 2 + items - 2 * (items // 2)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    i = 0
    for k in interesting_scores.keys():
        if k not in filtered_scores.keys():
            continue
        plt.subplot(rows, 2, i+1)
        plt.title(k, fontdict={'fontsize': 18})
        plt.tight_layout()
        binss = np.linspace(scores2[k].min(),
                            scores2[k].max(), num=15)   
        if len(set(binss)) == 1:
            binss = None
        if len(filtered_scores[k]) == scores2[k].count():
            ax = plt.hist(scores[k], bins=binss, alpha=1.0, color='g', label='All Scores')
        else:
            ax = plt.hist(scores[k], bins=binss, alpha=0.5, color='r', label='All Scores')
            plt.hist(filtered_scores[k], bins=binss, alpha=1.0, color='g', label='Filtered Scores')
        plt.hist(filtered_scores2[k], bins=binss, alpha=1.0, color='b', label='Filtered Scores2')
        if k != 'COO_HB' and interesting_scores[k] is not None:
            plt.plot((interesting_scores[k][0], interesting_scores[k][0]),
                     (0, max(ax[0])), c='k', linestyle=':')  # Adding line to show filter cut-off
        i += 1
    plt.savefig('filter_scores.png', dpi=300)


    # Copying filtered designs to separate directory
    filtered_dir = 'filtered_designs'
    if not os.path.exists(filtered_dir):
        os.mkdir(filtered_dir)
    
    for index, row in filtered_scores2.iterrows():
        name = row['description'] + '.pdb'
        copy2(f"{name}", f"{filtered_dir}/{name}")

if __name__ == "__main__":
    main()

