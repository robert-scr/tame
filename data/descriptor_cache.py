"""
Daten-Präparations-Skript für RDKit Deskriptoren des BACE-Datensatzes.

Dieses Skript:
1. Lädt den BACE-Datensatz via DeepChem
2. Berechnet alle verfügbaren RDKit 2D-Deskriptoren
3. Bereinigt & skaliert die Deskriptoren (Leakage-freie Imputation und Normalisierung)
4. Speichert die Ergebnisse als PyTorch tensors ab
"""

import os
import numpy as np
import torch

# DeepChem
import deepchem as dc
from deepchem.molnet import load_bace_classification

# RDKit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

# Sklearn für Imputation und Skalierung
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Unterdrücke RDKit Warnungen
RDLogger.DisableLog('rdApp.*')


def load_bace_smiles(splitter: str = "scaffold"):
    """
    Lade den BACE-Datensatz und extrahiere SMILES für Train/Valid/Test Split.
    
    Returns:
        train_smiles: List[str]
        valid_smiles: List[str]
        test_smiles: List[str]
    """
    print(f"Lade BACE-Datensatz via DeepChem ({splitter} split)...")
    
    tasks, datasets, transformers = load_bace_classification(
        featurizer='ECFP',
        splitter=splitter,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1
    )
    
    train_dataset, valid_dataset, test_dataset = datasets
    
    # Extrahiere SMILES-Strings (.ids enthält die SMILES)
    train_smiles = list(train_dataset.ids)
    valid_smiles = list(valid_dataset.ids)
    test_smiles = list(test_dataset.ids)
    
    print(f"  Train: {len(train_smiles)}")
    print(f"  Valid: {len(valid_smiles)}")
    print(f"  Test:  {len(test_smiles)}")
    
    return train_smiles, valid_smiles, test_smiles


def compute_descriptors(smiles_list, desc_names=None):
    """
    Berechne alle RDKit 2D-Deskriptoren für eine Liste von SMILES.
    
    Args:
        smiles_list: List[str]
        desc_names: Optional list of descriptor names. If None, uses all available.
    
    Returns:
        Array of shape (len(smiles_list), n_descriptors) with NaN for invalid SMILES.
    """
    if desc_names is None:
        # Alle verfügbaren RDKit Deskriptoren
        desc_names = [name for name, _ in Descriptors.descList]
    
    n_desc = len(desc_names)
    rows = []
    
    print(f"Berechne {n_desc} Deskriptoren für {len(smiles_list)} SMILES...")
    
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # Ungültige SMILES
                rows.append(np.full(n_desc, np.nan, dtype=np.float32))
                continue
            
            # Berechne alle Deskriptoren
            all_desc = Descriptors.CalcMolDescriptors(mol)
            row = []
            for name in desc_names:
                try:
                    val = all_desc.get(name, np.nan)
                    row.append(float(val))
                except Exception:
                    row.append(np.nan)
            
            rows.append(np.array(row, dtype=np.float32))
            
        except Exception as e:
            # Catch any RDKit exception
            rows.append(np.full(n_desc, np.nan, dtype=np.float32))
    
    arr = np.vstack(rows).astype(np.float32)
    print(f"  Shape nach Berechnung: {arr.shape}")
    
    return arr, desc_names


def clean_and_scale(
    train_desc,
    valid_desc,
    test_desc,
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    standardize: bool = True,
):
    """
    Bereinige, winsorize und skaliere Deskriptoren mit strikter Leakage-Prävention.
    
    Args:
        train_desc: Array of shape (N_train, n_features)
        valid_desc: Array of shape (N_valid, n_features)
        test_desc: Array of shape (N_test, n_features)
    
    Returns:
        train_cleaned, valid_cleaned, test_cleaned (winsorized and optionally scaled)
    """
    print("Bereinige, winsorize und skaliere Deskriptoren...")

    if not (0.0 <= winsor_lower < 1.0):
        raise ValueError("winsor_lower must be in [0, 1)")
    if not (0.0 < winsor_upper <= 1.0):
        raise ValueError("winsor_upper must be in (0, 1]")
    if winsor_lower >= winsor_upper:
        raise ValueError("winsor_lower must be < winsor_upper")
    
    # 1. Ersetze Inf/-Inf durch NaN
    print("  Ersetze Inf/-Inf durch NaN...")
    for arr in [train_desc, valid_desc, test_desc]:
        arr[~np.isfinite(arr)] = np.nan
    
    # 2. Imputation: Fit ONLY auf Training, transform auf alle 3
    print("  Imputation (median strategy)...")
    imputer = SimpleImputer(strategy='median')
    train_imputed = imputer.fit_transform(train_desc)
    valid_imputed = imputer.transform(valid_desc)
    test_imputed = imputer.transform(test_desc)
    
    # 3. Winsorizing: Fit ONLY auf Training, transform auf alle 3
    print(f"  Winsorizing (quantiles {winsor_lower:.3f} / {winsor_upper:.3f})...")
    train_low = np.quantile(train_imputed, winsor_lower, axis=0)
    train_high = np.quantile(train_imputed, winsor_upper, axis=0)
    train_winsorized = np.clip(train_imputed, train_low, train_high)
    valid_winsorized = np.clip(valid_imputed, train_low, train_high)
    test_winsorized = np.clip(test_imputed, train_low, train_high)

    # 4. Standardisierung: Fit ONLY auf Training, transform auf alle 3
    if standardize:
        print("  Standardisierung (StandardScaler)...")
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_winsorized)
        valid_scaled = scaler.transform(valid_winsorized)
        test_scaled = scaler.transform(test_winsorized)
    else:
        train_scaled = train_winsorized
        valid_scaled = valid_winsorized
        test_scaled = test_winsorized
    
    # Konvertiere zu float32
    train_scaled = train_scaled.astype(np.float32)
    valid_scaled = valid_scaled.astype(np.float32)
    test_scaled = test_scaled.astype(np.float32)
    
    print(f"  Finale Shapes:")
    print(f"    Train: {train_scaled.shape}")
    print(f"    Valid: {valid_scaled.shape}")
    print(f"    Test:  {test_scaled.shape}")
    
    return train_scaled, valid_scaled, test_scaled


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cache RDKit descriptors for BACE")
    parser.add_argument("--splitter", type=str, choices=["random", "scaffold"], default="scaffold",
                        help="BACE split strategy used to generate descriptor caches")
    args = parser.parse_args()

    # 1. Lade SMILES
    train_smiles, valid_smiles, test_smiles = load_bace_smiles(splitter=args.splitter)
    
    # 2. Berechne Deskriptoren
    train_desc, desc_names = compute_descriptors(train_smiles)
    valid_desc, _ = compute_descriptors(valid_smiles, desc_names=desc_names)
    test_desc, _ = compute_descriptors(test_smiles, desc_names=desc_names)
    
    # 3. Bereinigung & Skalierung
    train_scaled, valid_scaled, test_scaled = clean_and_scale(train_desc, valid_desc, test_desc)
    
    # 4. Konvertiere zu PyTorch Tensoren
    print("Konvertiere zu PyTorch tensors...")
    train_tensor = torch.from_numpy(train_scaled).to(torch.float32)
    valid_tensor = torch.from_numpy(valid_scaled).to(torch.float32)
    test_tensor = torch.from_numpy(test_scaled).to(torch.float32)
    
    # 5. Speichere ab
    output_dir = os.path.join(
        "C:\\Users\\robsc\\Home\\Dev\\molfusion2\\cache\\bace_RDKit_descriptors",
        args.splitter,
    )
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "bace_desc_train.pt")
    valid_path = os.path.join(output_dir, "bace_desc_valid.pt")
    test_path = os.path.join(output_dir, "bace_desc_test.pt")
    
    print(f"Speichere Tensoren in {output_dir}/...")
    torch.save(train_tensor, train_path)
    torch.save(valid_tensor, valid_path)
    torch.save(test_tensor, test_path)
    
    print("\n✓ Deskriptor-Caching abgeschlossen!")
    print(f"  {train_path}: {train_tensor.shape}")
    print(f"  {valid_path}: {valid_tensor.shape}")
    print(f"  {test_path}: {test_tensor.shape}")


if __name__ == "__main__":
    main()