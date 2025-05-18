from typing import Any, Dict, List
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, rdPartialCharges
import torch_geometric
from torch_geometric.data import Data


def estimate_pKa(mol):
    """
    Estimate the strongest acidic and basic pKa based on known functional groups.
    """
    if not mol:
        return None, None

    # Functional group-based pKa estimates (approximate values)
    pKa_values = {
        'carboxyl': 4.5, 'phenol': 10, 'sulfonic acid': 1, 'phosphate': 2.1,
        'amine': 9.5, 'imidazole': 6.9, 'guanidine': 12.5, 'thiol': 10.5
    }

    # Patterns for functional groups
    functional_groups = {
        'carboxyl': Chem.MolFromSmarts('[CX3](=O)[OX1H]'),
        'phenol': Chem.MolFromSmarts('c[OH]'),
        'sulfonic acid': Chem.MolFromSmarts('S(=O)(=O)[OH]'),
        'phosphate': Chem.MolFromSmarts('P(=O)([OH])([OH])O'),
        'amine': Chem.MolFromSmarts('[NX3;H2,H1]'),
        'imidazole': Chem.MolFromSmarts('c1[nH]cnc1'),
        'guanidine': Chem.MolFromSmarts('NC(=N)N'),
        'thiol': Chem.MolFromSmarts('[SX2H]')
    }

    strongest_acid_pKa = None
    strongest_base_pKa = None

    for group, pattern in functional_groups.items():
        if mol.HasSubstructMatch(pattern):
            pKa = pKa_values[group]
            if pKa < 7:
                strongest_acid_pKa = min(strongest_acid_pKa, pKa) if strongest_acid_pKa else pKa
            else:
                strongest_base_pKa = max(strongest_base_pKa, pKa) if strongest_base_pKa else pKa

    return strongest_acid_pKa, strongest_base_pKa


def estimate_pH(mol):
    """
    Estimate the pH range where the molecule is neutral.
    """
    acid_pKa, base_pKa = estimate_pKa(mol)

    if acid_pKa and base_pKa:
        return (acid_pKa + base_pKa) / 2  # Average of strongest acid and base
    elif acid_pKa:
        return acid_pKa - 1  # Below the acidic pKa
    elif base_pKa:
        return base_pKa + 1  # Above the basic pKa
    else:
        return 7  # Default neutral pH


def bin_value(value, bins):
    """Bins a continuous value into one-hot encoding."""
    bin_index = min(range(len(bins)), key=lambda i: abs(bins[i] - value))
    return F.one_hot(torch.tensor(bin_index), num_classes=len(bins))


feat_bins = {
    'TPSA': [0, 20, 40, 60, 80, 100, 120, 140, 160],  # 9
    'Fraction csp3': [0, 0.25, 0.5, 0.75, 1],  # 5
    'logP': [-3, -1, 1, 3, 5, 7],  # 6
    'pH': [0, 2, 4, 6, 8, 10, 12, 14],  # 8
    'Oxidation state': [-4, -2, 0, 2, 4, 6],  # 6
    'Molecular mass': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]  # 15
}

x_map: Dict[str, List[Any]] = {
    'Atomic number': list(range(0, 119)),
    'Chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'NDBN': list(range(0, 11)),
    'Formal charge': list(range(-5, 7)),
    'NBH': list(range(0, 9)),
    'NRE': list(range(0, 5)),
    'Hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'Aromatic': [False, True],
    'Ring': [False, True],

    # 'is_h_bond_donor': [False, True],
    # 'is_h_bond_acceptor': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


def from_rdmol(mol: Any) -> Data:
    assert isinstance(mol, Chem.Mol)

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    rdMolDescriptors.CalcOxidationNumbers(mol)  # Compute oxidation states

    # Calculate molecular descriptors and bin them
    tpsa = bin_value(Descriptors.TPSA(mol), feat_bins['TPSA'])
    fraction_csp3 = bin_value(Descriptors.FractionCSP3(mol), feat_bins['Fraction csp3'])
    logp = bin_value(Descriptors.MolLogP(mol), feat_bins['logP'])
    ph = bin_value(estimate_pH(mol), feat_bins['pH'])
    molecular_mass = bin_value(Descriptors.ExactMolWt(mol), feat_bins['Molecular mass'])

    xs: List[torch.Tensor] = []
    for atom in mol.GetAtoms():
        row: List[int] = []
        row.append(x_map['Atomic number'].index(atom.GetAtomicNum()))
        row.append(x_map['Chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['NDBN'].index(atom.GetTotalDegree()))
        row.append(x_map['Formal charge'].index(atom.GetFormalCharge()))
        row.append(x_map['NBH'].index(atom.GetTotalNumHs()))
        row.append(x_map['NRE'].index(atom.GetNumRadicalElectrons()))
        row.append(x_map['Hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['Aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['Ring'].index(atom.IsInRing()))

        oxidation_state_num = float(atom.GetProp("OxidationNumber"))
        oxidation_state = bin_value(oxidation_state_num, feat_bins['Oxidation state'])

        additional_one_hot_features = [tpsa, fraction_csp3, logp, ph, oxidation_state, molecular_mass]
        xs.append(torch.cat([F.one_hot(torch.tensor(i), num_classes=len(v)) for i, v in zip(row, x_map.values())] + additional_one_hot_features, dim=0))

    x = torch.stack(xs, dim=0).to(torch.float)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [torch.cat(
            [F.one_hot(torch.tensor(e_i), num_classes=len(e_map[key])) for e_i, key in zip(e, e_map.keys())],
            dim=0)] * 2

    edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
    edge_attr = torch.stack(edge_attrs, dim=0).to(torch.float)

    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def from_smiles(
    smiles: str,
    with_hydrogen: bool = False,
    kekulize: bool = False,
) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """

    RDLogger.DisableLog('rdApp.*')  # type: ignore

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    data = from_rdmol(mol)
    data.smiles = smiles
    return data


def convert_common_attributes_to_dict(common_attributes):
    features = ['Atomic number',
                'Chirality',
                'NDBN',
                'Formal charge',
                'NBH',
                'NRE',
                'Hybridization',
                'Aromatic',
                'Ring']
    additional_feat_bins = ['TPSA',
                            'Fraction csp3',
                            'logP',
                            'pH',
                            'Oxidation state',
                            'Molecular mass']

    features_dict = {}

    cur_start = 0
    for cur_feature in features:
        cur_feature_len = len(x_map[cur_feature])
        cur_importance = torch.sum(common_attributes[0, cur_start:cur_start+cur_feature_len]).item()
        cur_start += cur_feature_len
        features_dict[cur_feature] = [cur_importance]

    for cur_feature in additional_feat_bins:
        cur_feature_len = len(feat_bins[cur_feature])
        cur_importance = torch.sum(common_attributes[0, cur_start:cur_start+cur_feature_len]).item()
        cur_start += cur_feature_len
        features_dict[cur_feature] = [cur_importance]

    return features_dict


if __name__ == '__main__':
    pass