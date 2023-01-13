#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Indrek Kalvet
ikalvet@uw.edu
"""
import pyrosetta as pyr
import pyrosetta.rosetta
import os, sys
import numpy as np
import time
import argparse
import pandas as pd
import copy
import multiprocessing
from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t as xyzVector
from pyrosetta.rosetta.numeric import xyzMatrix_double_t
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.protocols.toolbox import superposition_transform
from pyrosetta.rosetta.protocols.toolbox import apply_superposition_transform
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.pose import append_subpose_to_pose

import design_utils
import scoring_utils
from scoring_utils import calculate_ddg



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
    return np.degrees(np.arctan2(y, x))


def match_sidechain_atoms(target, target_res_to_match,
                          target_atom_list_to_match, loop, loop_res_to_match,
                          loop_res_atom_list_to_match):
    """
    Given a pose of the target+a residue, and a pose of a loop, 
    and which residue to match

    Moves the loop to the stub and returns the overlayed pose

    returns a pose that has the loop in the last chain,
    and the target in other chains...
    Arguments:
        target (obj, pose) :: pose object -> metal thing
        target_res_to_match (int) :: rosetta residue index -> which cysetine you want to match
        target_atom_list_to_match (list) :: python array of atoms names ['SG', 'CB', 'CA']
        loop (obj, pose) :: pose of your macrocycle
        loop_res_to_match (int) :: Cys residue # in macrocycles
        loop_res_atom_list_to_match (list) :: python array of atoms names ['SG', 'CB', 'CA']

    Written by Patrick Salveson, provided by Meerit Said
    """

    # store the lengths of the two poses
    target_length = len(target.residues)
    loop_length = len(loop.residues)

    # make the rosetta object to hold xyz crds of the residue we want to align
    target_match_coords = xyzVector()
    loop_coords = xyzVector()

    # add the coords of the residues to be aligned to their respctive holders
    for target_name, loop_name in zip(target_atom_list_to_match, loop_res_atom_list_to_match):
        target_match_coords.append(
            target.residues[target_res_to_match].xyz(target_name))
        loop_coords.append(loop.residues[loop_res_to_match].xyz(loop_name))

    # make the rotation matrix and pose center rosetta objects
    rotation_matrix = xyzMatrix_double_t()
    target_center = xyzVector_double_t()
    loop_center = xyzVector_double_t()

    superposition_transform(
        loop_coords,
        target_match_coords,
        rotation_matrix,
        loop_center,
        target_center)

    apply_superposition_transform(
        loop,
        rotation_matrix,
        loop_center,
        target_center)

    # at this point "loop" is super imposed on the res in target
    # loop.dump_pdb('name.pdb')
    #########################################
    new_loop = Pose()
    new_loop.assign(loop)

    # at this point, the two objects are aligned
    # create a new empy pose object
    # splice the poses together and return
    spliced_pose = Pose()
    append_subpose_to_pose(spliced_pose, target, 1, target.size() - 1)
    append_subpose_to_pose(spliced_pose, loop, 1, loop_length)
    ###############################################################

    return spliced_pose


def replace_ligand_in_pose(pose, residue, resno, ref_atoms, new_atoms):
    _tmp_ligpose = pyrosetta.rosetta.core.pose.Pose()
    _tmp_ligpose.append_residue_by_jump(residue, 0)
    new_pose = match_sidechain_atoms(pose, resno, ref_atoms,
                                     _tmp_ligpose, 1, new_atoms)
    return new_pose


def get_rotamers_for_res_in_pose(in_pose, target_residue, sfx, ex1=True,
                                 ex2=True, ex3=False, ex4=False,
                                 check_clashes=True):
    packTask = pyrosetta.rosetta.core.pack.task.TaskFactory.create_packer_task(in_pose)
    packTask.set_bump_check(check_clashes)
    resTask = packTask.nonconst_residue_task(target_residue)
    resTask.or_ex1(ex1)
    resTask.or_ex2(ex2)
    resTask.or_ex3(ex3)
    resTask.or_ex4(ex4)
    resTask.restrict_to_repacking()
    packer_graph = pyrosetta.rosetta.utility.graph.Graph(in_pose.size())
    rsf = pyrosetta.rosetta.core.pack.rotamer_set.RotamerSetFactory()
    rotset = rsf.create_rotamer_set(in_pose)
    rotset.set_resid(target_residue)
    rotset.build_rotamers(in_pose, sfx, packTask, packer_graph, False)
    # print("Found num_rotamers:", rotset.num_rotamers())
    rotamer_set = []
    for irot in range(1, rotset.num_rotamers() + 1):
        rotamer_set.append(rotset.rotamer(irot).clone())
    return rotamer_set


def find_ligand_seqpos(pose):
    ligand_seqpos = None
    for res in pose.residues:
        if res.is_ligand() and not res.is_virtual_residue():
            ligand_seqpos = res.seqpos()
    return ligand_seqpos


def set_residue_random_rotamer(pose, resno, scorefxn):
    rotset = get_rotamers_for_res_in_pose(pose, resno, scorefxn)
    rotamer = rotset[np.random.randint(0, len(rotset))]
    for n in range(pose.residue(resno).nchi()):
        pose.residue(resno).set_chi(n+1, rotamer.chi(n+1))
    return pose


def fix_scorefxn(sfxn, allow_double_bb=False):
    opts = sfxn.energy_method_options()
    opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    opts.hbond_options().bb_donor_acceptor_check(not allow_double_bb)
    sfxn.set_energy_method_options(opts)


"""
Parsing user input
"""
def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB file, containing a ligand and matcher CST lines in header.")
    parser.add_argument("--nproc", type=int, default=os.cpu_count(), help="How CPU cores will be used?")
    parser.add_argument("--ligand", type=str, required=True, help="Ligand name that will be used for analysis")
    parser.add_argument("--params", type=str, nargs="+", required=True, help="Params files of ligands and noncanonicals")

    args = parser.parse_args()
    
    pdbfile = args.pdb
    ligand = args.ligand
    assert ligand in ["HRR", "HSS", "HSR", "HRS"]

    if 'SLURM_CPUS_PER_TASK' in os.environ.keys():
        NPROC = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        NPROC = args.nproc


    """
    Getting Rosetta started
    """
    params = args.params

    extra_res_fa = '-extra_res_fa'
    for p in params:
        if os.path.exists(p):
            extra_res_fa += " {}".format(p)

    pyr.init(f"-beta {extra_res_fa} -run:preserve_header -mute all")

    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex1:level", 1)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex2:level", 1)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex3:level", 1)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex4:level", 0)

    scorefxn = pyr.get_fa_scorefxn()
    sf = scorefxn.clone()
    fix_scorefxn(sf)


    """
    Loading the input PDB
    """
    pose = pyr.pose_from_file(pdbfile)

    matched_residues = design_utils.get_matcher_residues(pdbfile)
    his_seqpos = list(matched_residues.keys())[0]
    assert pose.residue(his_seqpos).name3() in ["HIS", "CYS", "CYX"], f"Bad coordinating residue: {pose.residue(his_seqpos).name3()}-{his_seqpos}"

    if ligand is not None:
        for p in params:
            if ligand in p:
                ligand_pdb = p.replace('.params', '.pdb')
                ligand_pose = pyr.pose_from_file(ligand_pdb)

    ligand_seqpos = find_ligand_seqpos(pose)
    existing_ligand = pose.residue(ligand_seqpos).name3()


    """
    Setting up things for alignment and getting the rotamers
    """
    ref_atoms = ['FE1', 'N1', 'N2', 'N3', 'N4']
    new_atoms = ['FE1', 'N1', 'N2', 'N3', 'N4']
    if existing_ligand in ["HRS", "HSR"] and ligand not in ["HSR", "HRS"]:
        ref_atoms = ['FE1', 'N4', 'N1', 'N2', 'N3']
    elif existing_ligand not in ["HRS", "HSR"] and ligand in ["HSR", "HRS"]:
        new_atoms = ['FE1', 'N4', 'N1', 'N2', 'N3']

    substrate_rotation_atoms = ["C28", "FE1", "C35", "C39"]
    # prop_chis = json.loads(open("/home/ikalvet/projects/Heme/theozyme/heme_propionate_chis.json", "r").read())

    # Getting all of the ligand rotamers
    rotset_HXX = get_rotamers_for_res_in_pose(ligand_pose, 1, scorefxn)

    if len(rotset_HXX) > 60:
        chis = {}
        for i, rotamer in enumerate(rotset_HXX):
            rot = get_dihedral(*[rotamer.xyz(a) for a in substrate_rotation_atoms])
            rot = round(rot, 0)
            if rot not in chis.keys():
                chis[rot] = []
            chis[rot].append(i)

        print(f"Found {len(chis)} substrate rotation chis: {chis.keys()}")

        # Picking 5 random rotamers for each chi bin
        _subsampled_rotamers = []
        for chi in chis:
            for n in range(5):
                _subsampled_rotamers.append(rotset_HXX[chis[chi][np.random.randint(0, len(chis[chi]))]])
        rotset_HXX = _subsampled_rotamers


    """
    Setting up design/repack layers
    """
    heavyatoms = design_utils.get_ligand_heavyatoms(pose)

    # Finding out what residues belong to what layer, based on the CA distance
    # from ligand heavyatoms.
    SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues\
        = design_utils.get_layer_selections(pose, [],
                                            [], ligand_seqpos, heavyatoms)

    pocket_residues = copy.deepcopy(residues[0])

    repack_residues = residues[1] + residues[2] + residues[3] + [ligand_seqpos]
    do_not_touch_residues = residues[4]

    for res in matched_residues:
        if res in pocket_residues:
            pocket_residues.__delitem__(pocket_residues.index(res))
            repack_residues.append(res)

    unclassified_residues = [res.seqpos() for res in pose.residues if res.seqpos() not in pocket_residues+repack_residues+do_not_touch_residues]
    assert len(unclassified_residues) == 0, f"Some residues have not been layered: {unclassified_residues}"

    # Saving no-ligand-repack residues.
    # Basically a list of all residues that are close-ish to the ligand.
    nlr_repack_residues = pocket_residues + repack_residues

    repack_res = ','.join([str(p) for p in repack_residues])
    do_not_touch_res = ','.join([str(p) for p in do_not_touch_residues])
    nlr_repack_res = ','.join([str(p) for p in nlr_repack_residues])


    xml_script = f"""
    <ROSETTASCRIPTS>  
      <SCOREFXNS>
            <ScoreFunction name="sfxn_clean" weights="beta">
                <Reweight scoretype="arg_cation_pi" weight="3"/>
            </ScoreFunction>
      </SCOREFXNS>

      <RESIDUE_SELECTORS>
          <Layer name="init_core_SCN" select_core="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
          <Layer name="init_boundary_SCN" select_boundary="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
          <Layer name="surface_SCN" select_surface="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
          <Layer name="coreRes" select_core="true" use_sidechain_neighbors="true" core_cutoff="2.1" surface_cutoff="1.0"/>

          <Index name="cat_residues" resnums="{his_seqpos}"/>
          <Index name="cat_his"      resnums="{his_seqpos}"/>

          <Index name="ligand_idx" resnums="{ligand_seqpos}"/>
          <Index name="repack_idx" resnums="{repack_res}"/>
          <Index name="do_not_touch_idx" resnums="{do_not_touch_res}"/>
          <Index name="nlr_repack_idx" resnums="{nlr_repack_res}"/>

      </RESIDUE_SELECTORS>

      <TASKOPERATIONS>
          <PruneBuriedUnsats name="prune_buried_unsats" allow_even_trades="false" atomic_depth_cutoff="3.5" minimum_hbond_energy="-0.5" />
          <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
          <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" />
          <SetCatalyticResPackBehavior name="catpack" fix_catalytic_aa="0" />
          <IncludeCurrent name="ic"/>

          <OperateOnResidueSubset name="repack_extended" selector="nlr_repack_idx">
              <RestrictToRepackingRLT/>
          </OperateOnResidueSubset>

          <OperateOnResidueSubset name="only_repack" selector="repack_idx">
              <RestrictToRepackingRLT/>
          </OperateOnResidueSubset>

          <OperateOnResidueSubset name="do_not_touch" selector="do_not_touch_idx">
              <PreventRepackingRLT/>
          </OperateOnResidueSubset>

      </TASKOPERATIONS>

      <MOVERS>
          <FastRelax name="fastRelax" scorefxn="sfxn_clean" repeats="1" task_operations="ex1_ex2aro,ic,do_not_touch,only_repack"/>
      </MOVERS>

    <PROTOCOLS>
        <Add mover="fastRelax"/>
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """


    # Creating an XML object from string.
    # This allows me to extract movers and stuff from the object,
    # and apply them separately.
    obj = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(xml_script)

    sfx = obj.get_score_function("sfxn_clean")
    fastRelax = obj.get_mover("fastRelax")
    fastRelax.constrain_relax_to_start_coords(True)

    rotscores = pd.DataFrame(columns=["score", "ddg", "rot"])
    
    print(f"Sampling {len(rotset_HXX)} rotamers for {ligand}")
    
    the_queue = multiprocessing.Queue()  # Queue stores the iterables
    manager = multiprocessing.Manager() 
    
    results = manager.dict()  # Need a special dictionary to store outputs from multiple processes

    for i, _r in enumerate(rotset_HXX):
        the_queue.put(i)
        results[i] = manager.dict()
    start = time.time()


    def process(q):
        while True:
            i = q.get(True)
            if i is None:
                return
            rotamer = rotset_HXX[i]
            repacked = {}

            for n in range(1):
                repacked[n] = replace_ligand_in_pose(pose, rotamer, ligand_seqpos, ref_atoms, new_atoms)

                # Randomly distorting the pocket residue conformers
                # to find alternative minima
                for r in pocket_residues:
                    repacked[n] = set_residue_random_rotamer(repacked[n], r, scorefxn)

                fastRelax.apply(repacked[n])

                sfx(repacked[n])

            minscore = 10000.0
            min_id = 0
            for n in repacked:
                if repacked[n].scores['total_score'] < minscore:
                    min_id = n
                    minscore = repacked[n].scores['total_score']

            # Get the ligand ddg, without including serine-ligand repulsion
            pose3 = repacked[min_id].clone()    
            ddg = calculate_ddg(pose3, sf, his_seqpos)

            results[i]['score'] = sfx(repacked[min_id])
            results[i]['ddg'] = ddg
            results[i]['rot'] = get_dihedral(*[repacked[min_id].residue(ligand_seqpos).xyz(a) for a in substrate_rotation_atoms])
            results[i]['pose'] = repacked[min_id]


    pool = multiprocessing.Pool(NPROC, process, (the_queue, ))

    # None to end each process
    for _i in range(NPROC):
        the_queue.put(None)
        
    # Closing the queue and the pool
    the_queue.close()
    the_queue.join_thread()
    pool.close()
    pool.join()
    end = time.time()

    for i in results.keys():
        for k in results[i].keys():
            if k == 'pose':
                continue
            rotscores.at[i, k] = float(results[i][k])
    
    best_idx_score = rotscores.score.astype(float).idxmin()
    best_idx_ddg = rotscores.ddg.astype(float).idxmin()
    if best_idx_ddg == best_idx_score:
        results[best_idx_score]['pose'].dump_pdb(pdbfile.replace('.pdb', f"_{ligand}_{best_idx_score}.pdb"))
    else:
        results[best_idx_score]['pose'].dump_pdb(pdbfile.replace('.pdb', f"_{ligand}_{best_idx_score}.pdb"))
        results[best_idx_ddg]['pose'].dump_pdb(pdbfile.replace('.pdb', f"_{ligand}_{best_idx_ddg}.pdb"))

    scoring_utils.dump_scorefile(rotscores, pdbfile.replace(".pdb", f"_{ligand}.sc"))
    print(f"{pdbfile}: Analysis done in {(end - start):.2f} seconds.")


if __name__ == "__main__":
    main()
