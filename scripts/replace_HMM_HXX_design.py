#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Indrek Kalvet
ikalvet@uw.edu
"""
import pyrosetta as pyr
import pyrosetta.rosetta
import os
import sys
import argparse
import numpy as np
import time
import pandas as pd
import json
import pyrosetta.distributed.io
from pyrosetta.rosetta.utility import vector1_numeric_xyzVector_double_t as xyzVector
from pyrosetta.rosetta.numeric import xyzMatrix_double_t
from pyrosetta.rosetta.numeric import xyzVector_double_t
from pyrosetta.rosetta.protocols.toolbox import superposition_transform
from pyrosetta.rosetta.protocols.toolbox import apply_superposition_transform
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.pose import append_subpose_to_pose

import design_utils
import no_ligand_repack
import scoring_utils


def translate_to_target(pose, target_pose, resno, target_resno, atomname, target_atomname):
    pose2 = pose.clone()
    shift = pose.residue(resno).xyz(atomname) - target_pose.residue(target_resno).xyz(target_atomname)
    for n in range(pose2.residue(resno).natoms()):
        pose2.residue(resno).set_xyz(n+1, pose2.residue(resno).xyz(n+1) - shift)
    return pose


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
    print("Found num_rotamers:", rotset.num_rotamers())
    rotamer_set = []
    for irot in range(1, rotset.num_rotamers() + 1):
        rotamer_set.append(rotset.rotamer(irot).clone())
    return rotamer_set


def check_bb_clash(pose, resno):
    """
    Checks if any heavyatom in the defined residue clashes with any backbone
    atom in the pose
    check_bb_clash(pose, resno) -> bool
    Arguments:
        pose (obj, pose)
        resno (int)
    """

    # Calculating the center of mass of the target residue
    trgt = pose.residue(resno)
    target_com = xyzVector_double_t()
    for g in 'xyz':
        crd = np.average([a.xyz().__getattribute__(g) for a in trgt.atoms()])
        target_com.__setattr__(g, crd)


    # Selecting residues that have CA withing 14A of the center of the mass
    # of the target residue. This is to reduce the search space.
    nbrs = []
    for res in pose.residues:
        if not res.is_ligand():
            if (res.xyz('CA') - target_com).norm() < 14.0:
                nbrs.append(res)

    # Iterating over each heavyatom in the target and checking if it clashes
    # with any backbone atom of any of the neighboring residues
    LIMIT = 1.0
    clash = False
    for res_atom_no in range(1, trgt.natoms()+1):
        if not trgt.atom_is_hydrogen(res_atom_no):
            res_atom = trgt.atom(res_atom_no)
            for res in nbrs:
                if res.seqpos == resno:
                    continue
                for bb_no in res.all_bb_atoms():
                    dist = (res_atom.xyz() - res.xyz(bb_no)).norm()
                    if dist < LIMIT:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
    return clash


def find_ligand_seqpos(pose):
    ligand_seqpos = None
    for res in pose.residues:
        if res.is_ligand() and not res.is_virtual_residue():
            ligand_seqpos = res.seqpos()
    return ligand_seqpos


def get_residues_with_close_sc(pose, ref_atoms, residues=None, exclude_residues=None, cutoff=4.5):
    """
    """
    if residues is None:
        residues = [x for x in range(1, pose.size()+1)]
    if exclude_residues is None:
        exclude_residues = []

    close_ones = []
    for resno in residues:
        if resno in exclude_residues:
            continue
        if pose.residue(resno).is_ligand():
            continue
        res = pose.residue(resno)
        close_enough = False
        for atomno in range(1, res.natoms()):
            if res.atom_type(atomno).is_heavyatom():
                for ha in ref_atoms:
                    if (res.xyz(atomno) - pose.residue(pose.size()).xyz(ha)).norm() < cutoff:
                        close_enough = True
                        close_ones.append(resno)
                        break
                    if close_enough is True:
                        break
                if close_enough is True:
                    break
    return close_ones


def get_pose_with_random_rotamer(pose, rotset, alignment_atoms, his_seqpos, propionate_chis=None):
    """
    Picking a random rotamer the non-clashing ligand rotamer.
    Adjusting the propionate sidechain conformers to be same as in the parent.
    """
    ligand_seqpos = find_ligand_seqpos(pose)
    ligand = rotset[0].name3()
    bb_clash = True
    _tries = 0
    
    while bb_clash is True and _tries < 100:
        _tries += 1
        rot_id = np.random.randint(0, len(rotset))
        rotamer = rotset[rot_id]
    
        pose2 = replace_ligand_in_pose(pose, rotamer, ligand_seqpos,
                                       alignment_atoms[0], alignment_atoms[1])
        if propionate_chis is not None:
            for ref_chi, new_chi in zip(propionate_chis["HMM"], propionate_chis[ligand]):
                pose2.residue(ligand_seqpos).set_chi(int(new_chi), pose.residue(ligand_seqpos).chi(int(ref_chi)))

        bb_clash = check_bb_clash(pose2, ligand_seqpos)
        print(f"Tried rotamer {rot_id}: clash = {bb_clash}")

    # Fixing the REMARKS in the PDB and loading the pose again
    _str = pyrosetta.distributed.io.to_pdbstring(pose2)
    pdbff = _str.split("\n")

    new_pdb = []
    if "ATOM" in pdbff[0]:
        new_pdb.append(f"REMARK 666 MATCH TEMPLATE X {ligand}    0 MATCH MOTIF A HIS  {his_seqpos}  1  1               \n")
        for l in pdbff:
            new_pdb.append(l)
    else:
        for l in pdbff:
            new_pdb.append(l)
            if "HEADER" in l:
                new_pdb.append(f"REMARK 666 MATCH TEMPLATE X {ligand}    0 MATCH MOTIF A HIS  {his_seqpos}  1  1               \n")

    pose3 = pyrosetta.Pose()
    pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose3, "\n".join(new_pdb))
    return pose3


def fix_scorefxn(sfxn, allow_double_bb=False):
    opts = sfxn.energy_method_options()
    opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    opts.hbond_options().bb_donor_acceptor_check(not allow_double_bb)
    sfxn.set_energy_method_options(opts)


def main():

    """
    Parsing user input
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB file, containing a ligand and matcher CST lines in header.")
    parser.add_argument("--nstruct", type=int, default=1, help="How many design iterations?")
    parser.add_argument("--suffix", type=str, help="Suffix to be added to the end the output filename")
    parser.add_argument("--design_pos", type=int, nargs="+", help="Positions that will be redesigned.")
    parser.add_argument("--keep_pos", type=int, nargs="+", help="Positions that will be kept fixed. Repack is allowed.")
    parser.add_argument("--params", type=str, nargs="+", help="Params files of ligands and noncanonicals.")
    parser.add_argument("--cstfile", type=str, help="Rosetta matcher/enzdes constraintfile for the ligand that is used for redesign.")
    parser.add_argument("--random_seed", action="store_true", default=False, help="Should the design be started from a random ligand rotamer?")
    parser.add_argument("--detect_pocket", action="store_true", default=False, help="Figure out designable positions around the ligand algorithmically.")
    parser.add_argument("--ligand", type=str, required=True, help="Ligand name that will be used for design")
    
    args = parser.parse_args()
    
    pdbfile = args.pdb
    ligand = args.ligand
    suffix = args.suffix
    random_seed = args.random_seed
    NSTRUCT = args.nstruct
    detect_pocket = args.detect_pocket
    cstfile = args.cstfile

    if detect_pocket is False and args.design_pos is None:
        sys.exit("Need to provide either a list of designable positions with --design_pos flag, "
                 "or enable algorithmic pocket residue detection with --detect_pocket flag")

    # Doing some arguments post-processing
    assert ligand in ["HRR", "HSS", "HSR", "HRS"], f"Invalid ligand name: {ligand}"
    
    design_pos = []
    if args.design_pos is not None:
        design_pos = [x for x in args.design_pos]

    keep_pos = []
    if args.keep_pos is not None:
        keep_pos = [x for x in args.keep_pos]

    suffix = ""
    if args.suffix is not None:
        suffix = args.suffix + "_"


    scorefilename = "scorefile.txt"
    outname = f"{os.path.basename(pdbfile).replace('.pdb','')}_{ligand}_{suffix}DE"


    """
    Getting Rosetta started
    """
    params = args.params

    extra_res_fa = '-extra_res_fa'
    for p in params:
        if os.path.exists(p):
            extra_res_fa += f" {p}"

    DAB = None
    DAB = "/home/ikalvet/Rosetta/DAlphaBall.gcc"
    
    assert DAB is not None, "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n"\
                            "For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"

    pyr.init(f"-dalphaball {DAB} {extra_res_fa} -beta_nov16 -run:preserve_header")

    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex1:level", 2)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex2:level", 2)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex3:level", 1)
    pyrosetta.rosetta.basic.options.set_integer_option("packing:ex4:level", 1)

    scorefxn = pyr.get_fa_scorefxn()


    """
    Loading the input PDB
    """
    pose = pyr.pose_from_file(pdbfile)

    if ligand is not None:
        for p in params:
            if ligand in p:
                ligand_pdb = p.replace('.params', '.pdb')
                ligand_pose = pyr.pose_from_file(ligand_pdb)

    matched_residues = design_utils.get_matcher_residues(pdbfile)
    his_seqpos = list(matched_residues.keys())[0]
    assert pose.residue(his_seqpos).name3() in ["HIS", "CYS", "CYX"], f"Bad coordinating residue: {pose.residue(his_seqpos).name3()}-{his_seqpos}\n"

    posex = pose.clone()

    ligand_seqpos = find_ligand_seqpos(pose)

    """
    Setting up things for alignment and getting the rotamers
    """
    # Getting all of the ligand rotamers
    ref_atoms = ['FE1', 'N1', 'N2', 'N3', 'N4']
    new_atoms = ['FE1', 'N1', 'N2', 'N3', 'N4']

    if ligand in ["HSR", "HRS"]:
        new_atoms = ['FE1', 'N4', 'N1', 'N2', 'N3']

    # Loading heme propionate chi definition information for all of the different ligands
    # This dictionary maps the chi numbers and atom names for each ligand in the same order
    prop_chis = json.loads(open(f"{os.path.dirname(os.path.realpath(__file__))}/../theozyme/heme_propionate_chis.json", "r").read())

    rotset_HXX = get_rotamers_for_res_in_pose(ligand_pose, 1, scorefxn)

    heme_atoms = ['FE1', 'N3', 'C21', 'C30', 'C19', 'N4', 'C14', 'C15', 'C18',
                  'C20', 'C16', 'C17', 'C31', 'C12', 'N1', 'C6', 'C7', 'C11',
                  'C13', 'C8', 'C9', 'C10', 'O2', 'O4', 'C28', 'C1', 'N2', 'C33',
                  'C32', 'C2', 'C3', 'C4', 'C5', 'O1', 'O3', 'C34', 'C29', 'C26',
                  'C25', 'C22', 'C23', 'C24', 'C27']

    substrate_atoms_ref = [pose.residue(ligand_seqpos).atom_name(n) for n in range(1, pose.residue(ligand_seqpos).natoms()+1) if pose.residue(ligand_seqpos).atom_name(n).strip() not in heme_atoms and pose.residue(ligand_seqpos).atom_type(n).element() != "H"]

    """
    Picking a random rotamer the non-clashing ligand rotamer.
    Adjusting the propionate sidechain conformers to be same as in the parent.
    """
    pose2 = get_pose_with_random_rotamer(posex, rotset_HXX, [ref_atoms, new_atoms], his_seqpos, prop_chis)


    """
    Setting up design/repack layers
    """
    heavyatoms = design_utils.get_ligand_heavyatoms(pose2)

    # Finding out what residues belong to what layer, based on the CA distance
    # from ligand heavyatoms.
    SEL_mutate_residues, SEL_repack_residues, SEL_do_not_repack, residues\
        = design_utils.get_layer_selections(pose2, keep_pos,
                                            design_pos, ligand_seqpos, heavyatoms)

    # design_residues = [5, 7, 8, 11, 12, 42, 43, 46, 39, 74, 75, 78, 109, 183]
    design_residues = design_pos
    if detect_pocket is True:
        # Detecting pocket residues algorithmically.
        # Only using ligand substrate atoms to find nearby residues.
        # If any positions have been set designable with --design_pos flag then that will be respected
        __a, __b, __c, residues_substrate\
            = design_utils.get_layer_selections(pose, keep_pos,
                                                design_pos, ligand_seqpos, substrate_atoms_ref, cuts=[6.0, 8.0, 10.0, 12.0])
        design_residues = [x for x in residues_substrate[0]+residues_substrate[1] if x not in matched_residues.keys()]
        design_residues += get_residues_with_close_sc(pose, substrate_atoms_ref, residues_substrate[2]+residues_substrate[3], matched_residues.keys(), 5.0)
        design_residues = list(set(design_residues))


    repack_residues = residues[2] + residues[3] + [ligand_seqpos]
    do_not_touch_residues = residues[4]

    for res in residues[0]+residues[1]:
        if res not in design_residues:
            repack_residues.append(res)

    repack_residues = [x for x in repack_residues if x not in design_residues]

    unclassified_residues = [res.seqpos() for res in pose.residues if res.seqpos() not in design_residues+repack_residues+do_not_touch_residues]
    assert len(unclassified_residues) == 0, f"Some residues have not been layered: {unclassified_residues}"

    # Saving no-ligand-repack residues.
    # Basically a list of all residues that are close-ish to the ligand.
    nlr_repack_residues = design_residues + repack_residues

    design_res = ','.join([str(x) for x in design_residues])
    repack_res = ','.join([str(p) for p in repack_residues])
    do_not_touch_res = ','.join([str(p) for p in do_not_touch_residues])
    nlr_repack_res = ','.join([str(p) for p in nlr_repack_residues + [ligand_seqpos]])

    print("Designable positions:", design_res.replace(",", "+"))


    xml_script = f"""
    <ROSETTASCRIPTS>  
      <SCOREFXNS>
            <ScoreFunction name="sfxn_design" weights="beta_nov16">
                <Reweight scoretype="res_type_constraint" weight="0.3"/>
                <Reweight scoretype="arg_cation_pi" weight="3"/>
                Reweight scoretype="approximate_buried_unsat_penalty" weight="5"/>
                <Reweight scoretype="aa_composition" weight="1.0" />
                Set approximate_buried_unsat_penalty_burial_atomic_depth="3.5"/>
                Set approximate_buried_unsat_penalty_hbond_energy_threshold="-0.5"/>
                Set approximate_buried_unsat_penalty_hbond_bonus_cross_chain="-1"/>

                for enzyme design
                <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                <Reweight scoretype="angle_constraint" weight="1.0"/>
                <Reweight scoretype="coordinate_constraint" weight="1.0"/>
            </ScoreFunction>

            <ScoreFunction name="sfxn_clean" weights="beta_nov16">
                <Reweight scoretype="arg_cation_pi" weight="3"/>
            </ScoreFunction>

            <ScoreFunction name="fa_csts" weights="beta_nov16">
                <Reweight scoretype="atom_pair_constraint" weight="1.0"/>
                <Reweight scoretype="dihedral_constraint" weight="1.0"/>
                <Reweight scoretype="angle_constraint" weight="1.0"/>
                <Reweight scoretype="coordinate_constraint" weight="1.0"/>
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
          
          <Index name="design_idx" resnums="{design_res}"/>
          <Index name="repack_idx" resnums="{repack_res}"/>
          <Index name="do_not_touch_idx" resnums="{do_not_touch_res}"/>
          <Index name="nlr_repack_idx" resnums="{nlr_repack_res}"/>

          <Chain name="chainA" chains="A"/>
          <Chain name="chainB" chains="B"/>

      </RESIDUE_SELECTORS>

      <TASKOPERATIONS>
          <RestrictAbsentCanonicalAAS name="noCys" keep_aas="ADEFGHIKLMNPQRSTVWY"/>
          <PruneBuriedUnsats name="prune_buried_unsats" allow_even_trades="false" atomic_depth_cutoff="3.5" minimum_hbond_energy="-0.5" />
          <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
          <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" />
          <SetCatalyticResPackBehavior name="catpack" fix_catalytic_aa="0" />
          <DetectProteinLigandInterface name="dpli" cut1="6.0" cut2="8.0" cut3="10.0" cut4="12.0" design="1" catres_only_interface="0" arg_sweep_interface="0" />
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

            <OperateOnResidueSubset name="disallow_MC" selector="design_idx">
                <RestrictAbsentCanonicalAASRLT aas="EDQNKRHSTYLIAVFWGP"/>
            </OperateOnResidueSubset>

          <ProteinInterfaceDesign name="pack_long" design_chain1="0" design_chain2="0" jump="1" interface_distance_cutoff="15"/>
      </TASKOPERATIONS>

      <MOVERS>
          <AddOrRemoveMatchCsts name="add_enz_csts" cstfile="{cstfile}" cst_instruction="add_new"/>

          <TaskAwareMinMover name="min" scorefxn="sfxn_clean" bb="0" chi="1" task_operations="pack_long" />
          <PackRotamersMover name="pack" scorefxn="sfxn_clean" task_operations="repack_extended,do_not_touch,ic,limitchi2,ex1_ex2aro"/>

          <FastRelax name="fastRelax" scorefxn="sfxn_clean" repeats="1" task_operations="ex1_ex2aro,ic"/>

          <FastDesign name="fastDesign" scorefxn="sfxn_design" repeats="1" task_operations="ex1_ex2aro,only_repack,do_not_touch,ic,limitchi2,disallow_MC" batch="false" ramp_down_constraints="false" cartesian="False" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"/> 

          <ClearConstraintsMover name="rm_csts" />

          <ScoreMover name="scorepose" scorefxn="sfxn_clean" verbose="false" />

      </MOVERS>

      <FILTERS>

          <ScoreType name="totalscore" scorefxn="sfxn_clean" threshold="9999" confidence="1"/>
          <ResidueCount name="nres" confidence="1" />
          <CalculatorFilter name="score_per_res" confidence="1" equation="SCORE/NRES" threshold="999">
              <Var name="SCORE" filter_name="totalscore" />
              <Var name="NRES" filter_name="nres" />
          </CalculatorFilter>

          <Geometry name="geom" count_bad_residues="true" confidence="0"/>

          <Ddg name="ddg_norepack"  threshold="0" jump="1" repeats="1" repack="0" relax_mover="min" confidence="0" scorefxn="sfxn_clean"/>	
          <Report name="ddg2" filter="ddg_norepack"/>

          <ShapeComplementarity name="sc" min_sc="0" min_interface="0" verbose="0" quick="0" jump="1" confidence="0"/>
          <ContactMolecularSurface name="contact_molecular_surface" distance_weight="0.5" target_selector="chainA" binder_selector="ligand_idx" confidence="0"/>

          <Sasa name="interface_buried_sasa" confidence="0" />
          <InterfaceHydrophobicResidueContacts name="hydrophobic_residue_contacts" target_selector="ligand_idx" binder_selector="chainA" scorefxn="sfxn_clean" confidence="0"/>

          <EnzScore name="all_cst" scorefxn="fa_csts" confidence="0" whole_pose="1" score_type="cstE" energy_cutoff="100.0"/>

      </FILTERS>

    <PROTOCOLS>
        <Add mover="add_enz_csts"/>
    </PROTOCOLS>
    </ROSETTASCRIPTS>
    """


    # run protocol and save output

    # Indrek: Using a different method the create an XML object from string.
    # This allows me to extract movers and stuff from the object,
    # and apply them separately.
    obj = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(xml_script)

    for N in range(1, NSTRUCT+1):
        """
        Indrek: I had to pull these movers out from the XML because some of the
        scoring did not work correctly when run within the XML.
        all_cst score always returned 0.0 for some reason.
        Also, this made testing much easier.
        """
        print(f"Design iteration {N} on pose: {os.path.basename(pdbfile)}")

        if os.path.exists(outname + f"_{N}.pdb"):
            print(f"Design {outname}_{N}.pdb already exists. Skipping iteration.")
            continue

        t0 = time.time()
        if random_seed is True:
            pose2 = get_pose_with_random_rotamer(posex, rotset_HXX, [ref_atoms, new_atoms], his_seqpos, prop_chis)

        pose3 = pose2.clone()

        # Fetching movers
        add_enz_csts = obj.get_mover("add_enz_csts")
        fastDesign_all_new = obj.get_mover("fastDesign")
        rm_csts = obj.get_mover("rm_csts")
        fastRelax = obj.get_mover("fastRelax")
        fastRelax.constrain_relax_to_start_coords(True)
        fastDesign_all_new.constrain_relax_to_start_coords(True)

        sfx = obj.get_score_function("sfxn_clean")
        sfx_cst = obj.get_score_function("sfxn_design")


        # Applying movers
        add_enz_csts.apply(pose3)
        _pose2 = pose3.clone()

        # Doing design iterations until the design is stable after relax
        DESIGN_LOOPS = 1

        design_poses = {}
        for N_des_loop in range(DESIGN_LOOPS):
            design_poses[N_des_loop] = pose3.clone()

            fastDesign_all_new.apply(design_poses[N_des_loop])

            pose3 = design_poses[N_des_loop].clone()


        # Fetching filters
        filters = {f: obj.get_filter(f) for f in obj.list_filters()}
        rm_csts.apply(pose3)

        # Applying filter scores to pose
        for fltr in filters:
            if fltr == "all_cst":
                design_utils.add_cst_wrapped_to_fix_bb(pose3, _pose2, add_enz_csts, cstfile, sfx_cst)
            scoring_utils.apply_score_from_filter(pose3, filters[fltr])
            if fltr == "all_cst":
                rm_csts.apply(pose3)

        for k in ['', 'defaultscorename']:
            try:
                pose3.scores.__delitem__(k)
            except KeyError:
                continue

        sfx(pose3)

        try:
            df_scores = pd.DataFrame.from_records([pose3.scores])
        except:
            print("The protocol failed. See log.")
            sys.exit(1)

        output_pdb_iter = outname + f"_{N}.pdb"
        pose3.dump_pdb(output_pdb_iter)

        #========================================================================
        # Extra filters
        #========================================================================

        df_scores['description'] = outname + f"_{N}"

        # Get the ligand ddg, without including serine-ligand repulsion
        from scoring_utils import calculate_ddg
        sf = pyr.get_fa_scorefxn()
        fix_scorefxn(sf)

        df_scores['corrected_ddg'] = calculate_ddg(pose3, sf, his_seqpos)

        # Calculating relative ligand SASA
        # First figuring out what is the path to the ligand PDB file

        free_ligand_sasa = scoring_utils.getSASA(ligand_pose, resno=1)
        ligand_sasa = scoring_utils.getSASA(pose3, resno=ligand_seqpos)
        df_scores.at[0, 'L_SASA'] = ligand_sasa / free_ligand_sasa


        # Using a custom function to find HBond partners of the COO groups.
        # The XML implementation misses some interations it seems.
        for n in range(1, 5):
            df_scores.at[0, f"O{n}_hbond"] = scoring_utils.find_hbonds_to_residue_atom(pose3, ligand_seqpos, f"O{n}")


        # Checking if both COO groups on heme are hbonded
        if any([df_scores.at[0, x] > 0.0 for x in ['O1_hbond', 'O3_hbond']]) and any([df_scores.at[0, x] > 0.0 for x in ['O2_hbond', 'O4_hbond']]):
            df_scores.at[0, 'COO_hbond'] = 1.0
        else:
            df_scores.at[0, 'COO_hbond'] = 0.0


        # Adding the constraints to the pose and using it to figure out
        # exactly which position belongs to which constraint
        _pose3 = pose3.clone()
        design_utils.add_cst_wrapped_to_fix_bb(_pose3, _pose2, add_enz_csts, cstfile, sfx_cst)
        target_resno = no_ligand_repack.get_target_residues_from_csts(_pose3)
        for i, resno in enumerate(target_resno):
            df_scores.at[0, f'SR{i+1}'] = resno

        # Calculating the no-ligand-repack as it's done in enzdes
        nlr_scores = no_ligand_repack.no_ligand_repack(_pose3, sfx, ligand_resno=ligand_seqpos)

        # Measuring the Heme-CYS/HIS angle because not all deviations
        # are reflected in the all_cst score
        heme_atoms = ["N1", "N2", "N3", "N4"]
        if pose3.residue(target_resno[0]).name3() == "CYX":
            target_atom = "SG"
        elif pose3.residue(target_resno[0]).name3() == "HIS":
            target_atom = "NE2"
        else:
            target_atom = None

        if target_atom is not None:
            angles = [scoring_utils.get_angle(pose3.residue(ligand_seqpos).xyz(nx),
                                              pose3.residue(ligand_seqpos).xyz("FE1"),
                                              pose3.residue(target_resno[0]).xyz(target_atom)) for nx in heme_atoms]
            df_scores.at[0, "heme_angle_wrst"] = min(angles)

        for k in nlr_scores.keys():
            df_scores.at[0, k] = nlr_scores.iloc[0][k]

        df_scores.at[0, 'subiteration'] = N_des_loop

        print(df_scores.iloc[0])
        scoring_utils.dump_scorefile(df_scores, scorefilename)

        t1 = time.time()
        print(f"Design iteration {N} took {(t1-t0):.3f} seconds and {N_des_loop} subiterations.")


if __name__ == "__main__":
    main()
