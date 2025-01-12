from rdkit import Chem
from rdkit.Chem import Fragments
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD, CalcTPSA, CalcFractionCSP3, CalcNumRotatableBonds
from rdkit.Chem.Lipinski import NumAromaticRings
import pandas as pd
import argparse
from pathlib import Path
import pubchempy as pcp
import pyscf
from pyscf.solvent.ddcosmo import DDCOSMO
from pyscf.data import radii
from pyscf.solvent.ddcosmo import ddcosmo_for_scf


def is_molecule_acceptable(molecule):
    allowed_elements = {'C', 'H', 'O', 'N', 'S'}
    allowed_functional_groups = [
        'amine', 'alcohol', 'carboxylic', 'sulfhydryl'
    ]

    functional_group_smarts = {
        'amine': '[NX3;!$(NC=O)]',
        'alcohol': '[OX2H]',
        'carboxylic': 'C(=O)[OX2H1]',
        'sulfhydryl': '[SX2H]'
    }

    undesired_functional_group_smarts = {
        'aldehyde': '[CX3H1](=O)[#6]',
        'ether': '[OD2]([#6])[#6]',
        'ester': '[CX3](=O)[OX2][#6]',
        'nitrile': '[NX1]#[CX2]',
        'thioether': '[SX2]([#6])[#6]',
        'azide': '[NX2][NX2]=[NX1]',
        'nitro': '[NX3](=O)[O-]',
        'sulfoxide': '[SX3](=O)[#6]',
        'sulfonic_acid': '[SX4](=O)(=O)[OX2H1]',
        'sulfone': '[SX4](=O)(=O)[#6]'
    }

    elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    element_counts = {el: elements.count(el) for el in set(elements)}

    if not set(element_counts.keys()).issubset(allowed_elements) or 'C' not in elements:
        return False

    if sum(element_counts.values()) > 40:
        return False

    functional_group_counts = {group: 0 for group in allowed_functional_groups}

    for group, smarts in functional_group_smarts.items():
        matches = molecule.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        functional_group_counts[group] += len(matches)

    if functional_group_counts['alcohol'] > 0 and functional_group_counts['carboxylic'] > 0 \
            and (functional_group_counts['alcohol'] - functional_group_counts['carboxylic'] > 0):
        return False
    elif functional_group_counts['sulfhydryl'] > 1:
        return False

    allowed_combinations = [
        {'amine'},
        {'amine', 'alcohol'},
        {'amine', 'carboxylic', 'alcohol'},
        {'amine', 'sulfhydryl'},
        {'carboxylic', 'alcohol'},
        {'carboxylic', 'sulfhydryl', 'alcohol'}
    ]

    detected_functional_groups = {group for group, count in functional_group_counts.items() if count > 0}

    if not any(detected_functional_groups == combo for combo in allowed_combinations):
        return False

    if 'carboxylic' in detected_functional_groups and 'sulfhydryl' in detected_functional_groups:
        if molecule.GetRingInfo().NumRings() > 0:
            return False

    for group, smarts in undesired_functional_group_smarts.items():
        if molecule.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return False

    # filter molecules containing "S-O" or "S=O" bonds
    sulfur_oxygen_smarts = [
        '[SX2]=[OX1]', '[SX4]=[OX1]', '[SX6]=[OX1]', '[SX2]=O', '[SX4]=O', '[SX6]=O',
        '[SX2][OX2]', '[SX4][OX2]', '[SX6][OX2]'
    ]
    for smarts in sulfur_oxygen_smarts:
        if molecule.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return False

    num_h_donors = rdMolDescriptors.CalcNumHBD(molecule)
    num_h_acceptors = rdMolDescriptors.CalcNumHBA(molecule)

    if num_h_donors > 2 or num_h_acceptors > 2:
        return False

    return True


def calculate_chemical_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {"SMILES": smiles, "Error": "Invalid SMILES"}

    properties = {}

    properties["Molecular Weight"] = Descriptors.ExactMolWt(mol)

    properties["Number of Hydrogen Donors"] = CalcNumHBD(mol)

    properties["Number of Hydrogen Acceptors"] = CalcNumHBA(mol)

    properties["Topological Polar Surface Area"] = CalcTPSA(mol)

    properties["FractionCsp3"] = CalcFractionCSP3(mol)

    properties["Number of Aromatic Rings"] = NumAromaticRings(mol)

    properties["Number of Rotatable Bonds"] = CalcNumRotatableBonds(mol)

    properties["Number of Heteroatoms"] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6))

    # Common electron-withdrawing groups: -NO2, -C≡N, -COOH
    num_accepting_groups = 0
    for substructure in ["[N+](=O)[O-]", "C#N", "C(=O)O"]:
        submol = Chem.MolFromSmarts(substructure)
        num_accepting_groups += len(mol.GetSubstructMatches(submol))
    properties["Number of Hole-Accepting Groups"] = num_accepting_groups

    properties["NHOH Count"] = Descriptors.NHOHCount(mol)

    properties["Number of Valence Electrons"] = sum(atom.GetTotalValence() for atom in mol.GetAtoms())

    return properties


def get_inchi_and_cid(smiles):
    try:
        # Use PubChem to search for the compound by SMILES
        compounds = pcp.get_compounds(smiles, 'smiles')

        if compounds:
            compound = compounds[0]
            # Retrieve the CID (PubChem Compound ID)
            cid = compound.cid
            inchi = compound.inchi

            return {
                "SMILES": smiles,
                "CID": cid,
                "InChI": inchi,
            }

        return {
            "SMILES": smiles,
            "CID": None,
            "InChI": None,
        }

    except Exception as e:
        return {
            "SMILES": smiles,
            "CID": f"Error: {e}",
            "InChI": f"Error: {e}",
        }


def _find_homo_lumo(mf):
    lumo = float("inf")
    lumo_idx = None
    homo = -float("inf")
    homo_idx = None
    for i, (energy, occ) in enumerate(zip(mf.mo_energy, mf.mo_occ)):
        if occ > 0 and energy > homo:
            homo = energy
            homo_idx = i
        if occ == 0 and energy < lumo:
            lumo = energy
            lumo_idx = i

    return homo, lumo


def get_rdkit_conformers(smiles, num_conformers=50, dielectric_constant=78.3553):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f'smiles: {smiles} is invalid!')
        exit()
    mol = Chem.AddHs(mol)
    embed_params = AllChem.ETKDGv3()
    AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=embed_params)

    mmf_params = AllChem.MMFFGetMoleculeProperties(mol)
    mmf_params.SetMMFFDielectricConstant(dielConst=dielectric_constant)
    AllChem.MMFFSanitizeMolecule(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, mmf_params, nonBondedThresh=500.0)
    res = AllChem.OptimizeMoleculeConfs(mol, ff)

    conformers = mol.GetConformers()
    energies = [e[1] for e in res]
    xyz_conformers = []
    tmps = []
    for conformer in conformers:
        num_atoms = mol.GetNumAtoms()
        xyz_format = f'{num_atoms}\n\n'
        tmp = ''
        for atom_idx in range(num_atoms):
            pos = conformer.GetAtomPosition(atom_idx)
            atom = mol.GetAtomWithIdx(atom_idx)
            symbol = atom.GetSymbol()
            x, y, z = pos.x, pos.y, pos.z
            xyz_format += f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"
            tmp += f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n"
        xyz_conformers.append(xyz_format)
        tmps.append(tmp)
    return xyz_conformers, energies, tmps


def get_homo_and_eox(smiles, basis=None, xc=None, cur_radii=None, radii_coef=None, dielectric_constant=78.3553,
                     a=None, b=None):
    xyz_conformers, energies, pyscf_conformers = get_rdkit_conformers(smiles, dielectric_constant=dielectric_constant)
    pyscf_conformers = sorted([(c, e) for c, e in zip(pyscf_conformers, energies)], key=lambda x: x[1])
    chosen_pyscf_conformer = pyscf_conformers[0][0]
    mol = pyscf.M(atom=chosen_pyscf_conformer, basis=basis, verbose=0)

    xyz_conformers = sorted([(c, e) for c, e in zip(xyz_conformers, energies)], key=lambda x: x[1])
    chosen_xyz_conformer = xyz_conformers[0][0]

    cosmo = DDCOSMO(mol)
    cosmo.eta = 0.1
    cosmo.lebedev_order = 19
    cosmo.eps = dielectric_constant
    cosmo.lmax = 6

    if cur_radii == 'MM3':
        cosmo.radii_table = radii.MM3 * radii_coef
    else:
        cosmo.radii_table = radii.UFF * radii_coef

    mf = mol.RKS(xc=xc)
    mf = ddcosmo_for_scf(mf, cosmo)
    mf.run()
    homo, lumo = _find_homo_lumo(mf)
    if homo != -float("inf"):
        homo_ev = homo * 27.2114
        eox = a * homo_ev + b if (a is not None and b is not None) else None
    else:
        homo_ev, eox = None, None
    return homo_ev, eox, chosen_xyz_conformer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process the path to a CSV file.")
    parser.add_argument('--input_smiles', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the input CSV file")
    args = parser.parse_args()
    data = pd.read_csv(args.input_smiles) # a CSV file with a single column of SMILES
    all_smiles = data['SMILES'].tolist()

    added_smiles = {}
    filtered_smiles = []

    for i, cur_smiles in enumerate(all_smiles):
        if '.' in cur_smiles:
            continue
        cur_mol = Chem.MolFromSmiles(cur_smiles)
        if cur_mol is None or not is_molecule_acceptable(cur_mol):
            continue
        else:
            to_add = Chem.MolToSmiles(cur_mol)
            if to_add not in added_smiles:
                filtered_smiles.append(to_add)
            added_smiles[to_add] = True

    print(f'filtered initial SMILES, {len(filtered_smiles)} molecules left.')

    data = []
    for cur_smiles in filtered_smiles:
        cur_row = {}
        cur_ids = get_inchi_and_cid(cur_smiles)
        cur_row.update(cur_ids)
        cur_props = calculate_chemical_properties(cur_smiles)
        cur_row.update(cur_props)

        for cur_key, cur_val in cur_row.items():
            if isinstance(cur_val, float):
                cur_row[cur_key] = round(cur_val, ndigits=3)

        data.append(cur_row)

    Path('./conformations').mkdir(exist_ok=True, parents=True)
    for i, cur_row in enumerate(data):
        cur_smiles = cur_row['SMILES']
        cur_homo, cur_eox, mol_conformation = get_homo_and_eox(cur_smiles, basis='cc-pvdz', xc='pbe0',
                                                               cur_radii='UFF', radii_coef=0.8,
                                                               dielectric_constant=78.3553,
                                                               a=-0.6613, b=-2.7729)

        with open(f'./conformations/{i}.xyz', 'w') as f:
            f.write(mol_conformation)

        cur_row['HOMO'] = round(cur_homo, ndigits=3)
        cur_row['Eox'] = round(cur_eox, ndigits=3)

    df = pd.DataFrame(data)
    df.to_csv(f'./{args.output_path}', index=False)
