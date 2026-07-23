from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
from rdkit.Chem import AllChem


def precompute_fingerprints(smiles_list, radius=2, fp_size=2048):
    """Precompute Morgan fingerprints for a list of SMILES strings."""
    fps = []
    for smiles in smiles_list:
        fps.append(get_morgan_fingerprint(smiles, rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)))
    return fps

def get_morgan_fingerprint(smiles, generator):
    """Modern approach - no deprecation warning"""
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return generator.GetFingerprint(mol)

def compute_tanimoto_similarity(fp1, fp2):
    """Compute Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def compute_similarity_vector(smiles, precomputed_fps, top_n=5, radius=2, fp_size=2048):
    """Compute Tanimoto similarity vector between a SMILES and a list of precomputed fingerprints."""
    query_fp = get_morgan_fingerprint(smiles, rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size))
    similarities = []
    for i, fp in enumerate(precomputed_fps):
        if fp is not None:
            similarities.append((i, compute_tanimoto_similarity(query_fp, fp)))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]