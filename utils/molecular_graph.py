from rdkit import Chem
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class MolGraph:
    """
    Graph container for speactral graph convolutions on molecular graphs.

    :param X: Node feature matrix of shape (n_nodes, n_node_features)
    :type X: np.ndarray
    :param edge_index: Edge index array of shape (2, n_edges)
    :type edge_index: np.ndarray
    :param edge_weight: Edge weight array of shape (n_edges,)
    :type edge_weight: np.ndarray
    :param n_nodes: Number of nodes in the graph
    :type n_nodes: int
    """
    X: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    n_nodes: int

    def build_adjacency(self) -> sp.csr_matrix:
        """
        Build the adjacency matrix from edge_index and edge_weight COO format into CSR format.
        
        :return: Adjacency matrix in CSR format
        :rtype: csr_matrix
        """
        i = self.edge_index[0]
        j = self.edge_index[1]
        data = self.edge_weight.astype(np.float32, copy=False)
        A = sp.coo_matrix((data, (i, j)), shape=(self.n_nodes, self.n_nodes)).tocsr()
        return A

    def build_normalized_laplacian(self) -> sp.csr_matrix:
        """
        Build the normalized graph Laplacian matrix L = I - D^{-1/2} A D^{-1/2}.
        
        :return: Normalized graph Laplacian matrix in CSR format
        :rtype: csr_matrix
        """
        A = self.build_adjacency()
        deg = np.asarray(A.sum(axis=1)).reshape(-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            deg_inv_sqrt = np.divide(1.0, np.sqrt(deg))
        deg_inv_sqrt[~np.isfinite(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(deg_inv_sqrt, format='csr')
        I = sp.eye(self.n_nodes, format='csr')
        L = I - D_inv_sqrt @ A @ D_inv_sqrt
        return L.tocsr()
    
    def build_scaled_laplacian(self, lambda_max: Optional[float] = 2.0) -> sp.csr_matrix:
        """
        Build the scaled graph Laplacian matrix L_scaled = (2 / lambda_max) * L - I,
        where lambda_max is the largest eigenvalue of L (estimated with eigsh).
        
        :param lambda_max: Largest eigenvalue of L. If None, it will be estimated. Defaults to 2.0.
        :type lambda_max: Optional[float]
        :return: Scaled graph Laplacian matrix in CSR format
        :rtype: csr_matrix
        """
        L = self.build_normalized_laplacian()
        if lambda_max is None:
            try:
                lambda_max = float(eigsh(L, k=1, which='LM', return_eigenvectors=False)[0])
            except Exception:
                lambda_max = 2.0
        if lambda_max <= 0:
            lambda_max = 2.0
        I = sp.eye(self.n_nodes, format='csr')
        L_scaled = np.divide(2.0, lambda_max) * L - I
        return L_scaled.tocsr()


def bond_weight(bond: Chem.Bond) -> float:
    """
    Helper function to get bond weight based on bond type.
    
    :param bond: Bond object from RDKit
    :type bond: Chem.Bond
    :return: Weight of the bond based on its type
    :rtype: float
    """
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        return 1.0
    if bt == Chem.rdchem.BondType.DOUBLE:
        return 2.0
    if bt == Chem.rdchem.BondType.TRIPLE:
        return 3.0
    if bt == Chem.rdchem.BondType.AROMATIC:
        return 1.5
    return 1.0


def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """
    Generate atome features for a given RDKit atom. 
    Features include one-hot encodings for atomic number, degree, hybridization,
    number of implicit hydrogens, aromaticity, and linear formal charge.
    
    :param atom: Atom object from RDKit
    :type atom: Chem.Atom
    :return: Numpy array of atom features
    :rtype: np.ndarray
    """
    atomic_number_choices = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
    degree_choices = [0, 1, 2, 3, 4, 5]
    hybridization_choices = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    number_Hs_choices = [0, 1, 2, 3, 4]

    z = atom.GetAtomicNum()
    n_H = atom.GetTotalNumHs()
    formal_charge = atom.GetFormalCharge()
    is_aromatic = atom.GetIsAromatic()
    hybridization = atom.GetHybridization()
    degree = atom.GetDegree()

    feats = []
    feats += _one_hot(z, atomic_number_choices)
    feats += _one_hot(degree, degree_choices)
    feats += _one_hot(hybridization, hybridization_choices)
    feats += _one_hot(n_H, number_Hs_choices)
    feats.append(1.0 if is_aromatic else 0.0)
    feats.append(formal_charge)

    return np.asarray(feats, dtype=np.float32)


def _one_hot(value: int, choices: List) -> List[float]:
    """
    Helper function to create a one-hot encoding for a given value based on a list of choices.
    
    :param value: Value to encode into one-hot format of length n
    :type value: int
    :param choices: List of possible choices for encoding of length n
    :type choices: List
    :return: One-hot encoded list of length n
    :rtype: List[float]
    """
    encoding = [0.0] * len(choices)
    if value in choices:
        index = choices.index(value)
        encoding[index] = 1.0
    return encoding


#TODO multiple molecules as block diagonal
def smiles_to_graph(smiles: str, add_hydrogens: bool = False) -> MolGraph:
    """
    Convert a SMILES string to a MolGraph object.
    
    :param smiles: SMILES string representing a molecule
    :type smiles: str
    :param add_hydrogens: Whether to add explicit hydrogen atoms. Default False
        for most tasks (heavy atoms only), set True for quantum property prediction.
    :type add_hydrogens: bool
    :return: MolGraph object representing the molecular graph
    :rtype: MolGraph
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f" >>> ERROR: Invalid SMILES: {smiles!r}")
    
    # Optionally add explicit hydrogens (important for quantum properties)
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    
    n_atoms = mol.GetNumAtoms()
    X = np.stack([get_atom_features(mol.GetAtomWithIdx(i)) for i in range(n_atoms)], axis = 0)
    
    rows: List[int] = []
    cols: List[int] = []
    wts: List[float] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        w = float(bond_weight(bond))

        # undirected -> store both directions
        rows += [i, j]
        cols += [j, i]
        wts += [w, w]

    edge_index = np.asarray([rows, cols], dtype=np.int64)
    edge_weight = np.asarray(wts, dtype=np.float32)

    return MolGraph(X=X, edge_index=edge_index, edge_weight=edge_weight, n_nodes=n_atoms)
