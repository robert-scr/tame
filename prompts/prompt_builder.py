"""In-Context Learning (ICL) prompt builders for molecular property prediction.

This module provides:
1. Task-specific ICL prompt templates
2. `ICLPromptBuilder` class with factory methods for different tasks
3. Few-shot example formatting utilities

Supported tasks:
- Solubility (ESOL): log S prediction
- Lipophilicity: LogP/LogD prediction  
- Quantum properties: HOMO, LUMO, gap, etc.
- Generic: Custom property prediction

Usage:
    >>> from prompts import ICLPromptBuilder
    >>> 
    >>> # Create builder for solubility task
    >>> builder = ICLPromptBuilder.for_solubility()
    >>> 
    >>> # Format few-shot examples
    >>> examples = [("CCO", -0.3), ("CCCC", -2.1)]
    >>> system, user = builder.create_prompt("c1ccccc1", examples=examples)
    >>> 
    >>> # Or use with similarity-based example selection
    >>> from utils.similarity import precompute_fingerprints, compute_similarity_vector
    >>> train_fps = precompute_fingerprints(train_smiles)
    >>> similar_examples = builder.select_similar_examples(
    ...     query_smiles="c1ccccc1",
    ...     train_smiles=train_smiles,
    ...     train_labels=train_labels,
    ...     train_fps=train_fps,
    ...     n_examples=10
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ICLPromptConfig:
    """Configuration for ICL prompt building.
    
    Attributes:
        task_name: Name of the prediction task
        property_name: Human-readable property name (for prompts)
        property_unit: Unit of the property (for prompts)
        value_range: Typical value range description
        system_prompt: System prompt template
        task_instruction: Task-specific instruction
        example_format: Format string for examples (with {smiles} and {value} placeholders)
        query_format: Format string for query (with {smiles} placeholder)
        include_similarity: Whether to include similarity in example format
    """
    task_name: str = "generic"
    property_name: str = "molecular property"
    property_unit: str = ""
    value_range: str = ""
    system_prompt: str = ""
    task_instruction: str = "Please strictly follow the format, no other information can be provided."
    example_format: str = "SMILES: '{smiles}' → {value:.2f}"
    query_format: str = "SMILES: '{smiles}' → "
    include_similarity: bool = False
    decimal_places: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "task_name": self.task_name,
            "property_name": self.property_name,
            "property_unit": self.property_unit,
            "value_range": self.value_range,
            "include_similarity": self.include_similarity,
            "decimal_places": self.decimal_places,
        }


# =============================================================================
# Pre-defined ICL Templates
# =============================================================================

# --- Solubility (ESOL) ---
SOLUBILITY_ICL_SYSTEM = """You are an expert chemist with deep knowledge in computational chemistry and molecular property prediction.
Your specialty is predicting aqueous solubility of organic molecules from their structure.

TASK: Predict the logarithm of aqueous solubility (log S) where S is measured in mol/L.

DOMAIN KNOWLEDGE:
- Solubility depends on molecular weight, hydrophobicity (LogP), hydrogen bonding, and molecular structure
- Polar functional groups (OH, NH2, COOH) generally increase solubility
- Aromatic rings and long alkyl chains decrease solubility
- Log S typically ranges from -11 (highly insoluble) to +1 (highly soluble)

OUTPUT FORMAT REQUIREMENTS:
- Provide ONLY a single numeric value
- Use log base 10
- More negative values indicate lower solubility
- Do NOT include units, explanations, reasoning, or any additional text"""

# --- Lipophilicity (LogP/LogD) ---
LIPOPHILICITY_ICL_SYSTEM = """You are an expert chemist with deep knowledge in computational chemistry and molecular property prediction.
Your specialty is predicting lipophilicity (partition coefficient) of organic molecules.

TASK: Predict the logarithm of the octanol-water partition coefficient (LogP or LogD).

DOMAIN KNOWLEDGE:
- LogP measures how a molecule partitions between octanol (lipophilic) and water (hydrophilic)
- Positive LogP indicates preference for lipophilic environment
- Negative LogP indicates preference for aqueous environment
- Aromatic rings, alkyl chains, and halogens increase LogP
- Polar groups (OH, NH2, COOH) and ionizable groups decrease LogP
- Drug-like molecules typically have LogP between -0.4 and 5.6 (Lipinski's rule)

OUTPUT FORMAT REQUIREMENTS:
- Provide ONLY a single numeric value
- Use log base 10
- Do NOT include units, explanations, reasoning, or any additional text"""

# --- Quantum Properties (HOMO, LUMO, Gap, etc.) ---
QUANTUM_ICL_SYSTEM = """You are an expert quantum chemist with deep knowledge in molecular electronic structure.
Your specialty is predicting quantum mechanical properties of organic molecules.

TASK: Predict the specified quantum property value.

DOMAIN KNOWLEDGE:
- HOMO (Highest Occupied Molecular Orbital) relates to ionization potential and electron-donating ability
- LUMO (Lowest Unoccupied Molecular Orbital) relates to electron affinity and electron-accepting ability
- HOMO-LUMO gap indicates chemical stability and reactivity
- Conjugated systems have smaller gaps; saturated molecules have larger gaps
- Electron-donating groups raise HOMO; electron-withdrawing groups lower LUMO
- Values are typically in electronvolts (eV) or Hartrees

OUTPUT FORMAT REQUIREMENTS:
- Provide ONLY a single numeric value
- Use the appropriate unit scale (typically eV)
- Do NOT include units, explanations, reasoning, or any additional text"""

# --- Generic Property ---
GENERIC_ICL_SYSTEM = """You are an expert chemist with deep knowledge in computational chemistry and molecular property prediction.
Your task is to predict molecular properties based on chemical structure.

TASK: Predict the specified molecular property value from the SMILES representation.

DOMAIN KNOWLEDGE:
- Molecular properties depend on functional groups, structure, and electronic effects
- Use pattern recognition from the provided examples
- Consider structural similarity to training examples

OUTPUT FORMAT REQUIREMENTS:
- Provide ONLY a single numeric value
- Match the format and scale of the provided examples
- Do NOT include units, explanations, reasoning, or any additional text"""

# --- Binding Affinity (BACE/pIC50) ---
BINDING_ICL_SYSTEM = """You are an expert medicinal chemist with deep knowledge in drug discovery and structure-activity relationships (SAR).
Your specialty is predicting binding affinity of small molecules to enzyme targets, particularly β-secretase (BACE-1).

TASK: Predict the pIC50 value (negative log of IC50 in molar units) for BACE-1 inhibition.

DOMAIN KNOWLEDGE:
- pIC50 = -log10(IC50) where IC50 is the half-maximal inhibitory concentration
- Higher pIC50 values indicate stronger binding/better inhibitors
- BACE-1 inhibitors typically contain:
  * Hydroxyethylamine or aminothiazine core scaffolds
  * Basic amines to interact with catalytic aspartates (Asp32, Asp228)
  * Aromatic/hydrophobic groups to fill S1, S2', S3 pockets
  * Hydrogen bond acceptors/donors for specificity
- pIC50 typically ranges from 3 (weak, ~1mM) to 10 (potent, ~0.1nM)
- Drug-like BACE-1 inhibitors usually have pIC50 > 6 (IC50 < 1μM)

STRUCTURAL CONSIDERATIONS:
- Presence of hydroxyethylamine → often pIC50 > 7
- Aminothiazine/aminooxazine scaffolds → good starting points
- Fluorinated aromatics → improved metabolic stability
- Macrocyclic constraints → enhanced potency and selectivity
- P-glycoprotein substrates may show reduced brain penetration

OUTPUT FORMAT REQUIREMENTS:
- Provide ONLY a single numeric value (the predicted pIC50)
- Typical range: 3.0 to 10.0
- Do NOT include units, explanations, reasoning, or any additional text"""


# --- Binding Classification (Binary Active/Inactive) ---
BINDING_CLASSIFICATION_ICL_SYSTEM = """You are an expert medicinal chemist with deep knowledge in drug discovery and structure-activity relationships (SAR).
Your specialty is predicting whether small molecules are active inhibitors of β-secretase (BACE-1).

TASK: Predict whether a molecule is ACTIVE (1) or INACTIVE (0) as a BACE-1 inhibitor.

DOMAIN KNOWLEDGE:
- BACE-1 is a key target for Alzheimer's disease therapy
- Active inhibitors typically have IC50 < 1μM (pIC50 > 6)
- Active BACE-1 inhibitors often contain:
  * Hydroxyethylamine or aminothiazine core scaffolds
  * Basic amines to interact with catalytic aspartates (Asp32, Asp228)
  * Aromatic/hydrophobic groups to fill S1, S2', S3 pockets
  * Hydrogen bond acceptors/donors for specificity
- Inactive molecules may lack key pharmacophore features or have poor fit to the binding site

STRUCTURAL CONSIDERATIONS FOR ACTIVITY:
- Presence of hydroxyethylamine → likely active
- Aminothiazine/aminooxazine scaffolds → good chance of activity
- Multiple aromatic rings with proper spacing → favorable
- Polar groups that can form H-bonds with catalytic site → important
- Very large or very small molecules → often inactive

OUTPUT FORMAT REQUIREMENTS:
- Provide ONLY a single number: 1 (active) or 0 (inactive)
- Do NOT include any explanations, reasoning, or additional text
- Just output the number 1 or 0"""


# =============================================================================
# ICL Prompt Builder
# =============================================================================

class ICLPromptBuilder:
    """
    In-Context Learning prompt builder for molecular property prediction.
    
    Handles:
    - Task-specific prompt templates
    - Few-shot example formatting
    - Similarity-based example selection
    - Zero-shot and few-shot modes
    
    Example:
        >>> from prompts import ICLPromptBuilder
        >>> 
        >>> # Using factory method
        >>> builder = ICLPromptBuilder.for_solubility()
        >>> 
        >>> # Create prompt with examples
        >>> examples = [("CCO", -0.3), ("CCCC", -2.1), ("c1ccccc1", -1.5)]
        >>> system, user = builder.create_prompt("CC(=O)O", examples=examples)
        >>> 
        >>> # Or with similarity-based selection
        >>> system, user = builder.create_prompt_with_similarity(
        ...     query_smiles="CC(=O)O",
        ...     train_smiles=train_smiles,
        ...     train_labels=train_labels,
        ...     train_fps=train_fps,
        ...     n_examples=10
        ... )
    """
    
    def __init__(self, config: Optional[ICLPromptConfig] = None) -> None:
        """
        Initialize ICL prompt builder.
        
        Args:
            config: ICLPromptConfig instance (if None, uses defaults)
        """
        self.config = config or ICLPromptConfig()
    
    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def for_solubility(cls, **kwargs) -> "ICLPromptBuilder":
        """Create an ICL builder configured for aqueous solubility (log S) prediction."""
        config = ICLPromptConfig(
            task_name="solubility",
            property_name="aqueous solubility",
            property_unit="log mol/L",
            value_range="-11 (insoluble) to +1 (highly soluble)",
            system_prompt=SOLUBILITY_ICL_SYSTEM,
            example_format="SMILES: '{smiles}' → {value:.2f}",
            query_format="SMILES: '{smiles}' → ",
            decimal_places=2,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_lipophilicity(cls, **kwargs) -> "ICLPromptBuilder":
        """Create an ICL builder configured for lipophilicity (LogP) prediction."""
        config = ICLPromptConfig(
            task_name="lipophilicity",
            property_name="lipophilicity (LogP)",
            property_unit="log P",
            value_range="-3 (hydrophilic) to +7 (lipophilic)",
            system_prompt=LIPOPHILICITY_ICL_SYSTEM,
            example_format="SMILES: '{smiles}' → {value:.2f}",
            query_format="SMILES: '{smiles}' → ",
            decimal_places=2,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_quantum(
        cls, 
        property_name: str = "HOMO-LUMO gap",
        value_range: str = "0 to 15 eV",
        decimal_places: int = 4,
        **kwargs
    ) -> "ICLPromptBuilder":
        """
        Create an ICL builder configured for quantum property prediction.
        
        Args:
            property_name: Name of quantum property (e.g., "HOMO", "LUMO", "gap")
            value_range: Typical value range
            decimal_places: Decimal precision for values
        """
        config = ICLPromptConfig(
            task_name="quantum",
            property_name=property_name,
            property_unit="eV",
            value_range=value_range,
            system_prompt=QUANTUM_ICL_SYSTEM,
            example_format="SMILES: '{smiles}' → {value:.4f}",
            query_format="SMILES: '{smiles}' → ",
            decimal_places=decimal_places,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_generic(
        cls,
        property_name: str = "property",
        property_unit: str = "",
        value_range: str = "",
        decimal_places: int = 2,
        **kwargs
    ) -> "ICLPromptBuilder":
        """
        Create an ICL builder for generic property prediction.
        
        Args:
            property_name: Name of the property
            property_unit: Unit of measurement
            value_range: Typical value range description
            decimal_places: Decimal precision for values
        """
        config = ICLPromptConfig(
            task_name="generic",
            property_name=property_name,
            property_unit=property_unit,
            value_range=value_range,
            system_prompt=GENERIC_ICL_SYSTEM,
            example_format="SMILES: '{smiles}' → {value:.2f}",
            query_format="SMILES: '{smiles}' → ",
            decimal_places=decimal_places,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_binding_affinity(cls, **kwargs) -> "ICLPromptBuilder":
        """Create an ICL builder configured for binding affinity (pIC50) prediction.
        
        Optimized for BACE-1 inhibition data, but can be used for other targets.
        Uses domain knowledge about enzyme inhibition and SAR.
        """
        config = ICLPromptConfig(
            task_name="binding",
            property_name="BACE-1 binding affinity",
            property_unit="pIC50",
            value_range="3.0 (weak, ~1mM IC50) to 10.0 (potent, ~0.1nM IC50)",
            system_prompt=BINDING_ICL_SYSTEM,
            example_format="SMILES: '{smiles}' → {value:.2f}",
            query_format="SMILES: '{smiles}' → ",
            decimal_places=2,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_binding_classification(cls, **kwargs) -> "ICLPromptBuilder":
        """Create an ICL builder for binary BACE-1 activity classification.
        
        Predicts 0 (inactive) or 1 (active) instead of continuous pIC50 values.
        Use this for the BACE classification dataset.
        """
        config = ICLPromptConfig(
            task_name="binding_classification",
            property_name="BACE-1 activity",
            property_unit="",
            value_range="0 (inactive) or 1 (active)",
            system_prompt=BINDING_CLASSIFICATION_ICL_SYSTEM,
            example_format="SMILES: '{smiles}' → {value:.0f}",  # Integer format
            query_format="SMILES: '{smiles}' → ",
            decimal_places=0,  # Binary: 0 or 1
            **kwargs
        )
        return cls(config=config)
    
    # -------------------------------------------------------------------------
    # Prompt Construction
    # -------------------------------------------------------------------------
    
    def format_example(
        self, 
        smiles: str, 
        value: float, 
        similarity: Optional[float] = None
    ) -> str:
        """
        Format a single example for the prompt.
        
        Args:
            smiles: SMILES string
            value: Property value
            similarity: Optional Tanimoto similarity (if include_similarity=True)
            
        Returns:
            Formatted example string
        """
        if self.config.include_similarity and similarity is not None:
            return f"SMILES: '{smiles}' → {value:.{self.config.decimal_places}f} (Similarity: {similarity:.3f})"
        else:
            fmt = self.config.example_format
            return fmt.format(smiles=smiles, value=value)
    
    def format_examples(
        self, 
        examples: List[Tuple[str, float, Optional[float]]]
    ) -> str:
        """
        Format multiple examples for the prompt.
        
        Args:
            examples: List of (smiles, value) or (smiles, value, similarity) tuples
            
        Returns:
            Formatted examples string
        """
        lines = []
        for ex in examples:
            if len(ex) == 3:
                smiles, value, sim = ex
            else:
                smiles, value = ex
                sim = None
            lines.append(self.format_example(smiles, value, sim))
        
        return "Examples:\n" + "\n".join(lines)
    
    def create_prompt(
        self,
        smiles: str,
        examples: Optional[List[Tuple[str, float]]] = None,
    ) -> Tuple[str, str]:
        """
        Create ICL prompt for a molecule.
        
        Args:
            smiles: Query SMILES string
            examples: Optional list of (smiles, value) tuples for few-shot
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        query = self.config.query_format.format(smiles=smiles)
        
        if examples is not None and len(examples) > 0:
            examples_str = self.format_examples(examples)
            user_content = f"{self.config.task_instruction}\n\n{examples_str}\n\n{query}"
        else:
            user_content = f"{self.config.task_instruction}\n\n{query}"
        
        return self.config.system_prompt, user_content
    
    # -------------------------------------------------------------------------
    # Similarity-Based Example Selection
    # -------------------------------------------------------------------------
    
    def select_similar_examples(
        self,
        query_smiles: str,
        train_smiles: List[str],
        train_labels: Union[List[float], np.ndarray],
        train_fps: List,
        n_examples: int = 10,
    ) -> List[Tuple[str, float, float]]:
        """
        Select most similar training examples for few-shot prompting.
        
        Args:
            query_smiles: Query molecule SMILES
            train_smiles: List of training SMILES
            train_labels: Training labels (values)
            train_fps: Pre-computed fingerprints from precompute_fingerprints()
            n_examples: Number of examples to select
            
        Returns:
            List of (smiles, value, similarity) tuples, sorted by similarity (descending)
        """
        try:
            from utils.similarity import precompute_fingerprints, compute_similarity_vector
        except ImportError:
            raise ImportError("utils.similarity module required for similarity-based selection")
        
        # Get similar molecules
        similar_indices = compute_similarity_vector(query_smiles, train_fps, top_n=n_examples)
        
        # Convert labels to numpy if needed
        labels = np.asarray(train_labels)
        
        examples = []
        for idx, sim in similar_indices:
            examples.append((train_smiles[idx], float(labels[idx]), sim))
        
        return examples
    
    def create_prompt_with_similarity(
        self,
        query_smiles: str,
        train_smiles: List[str],
        train_labels: Union[List[float], np.ndarray],
        train_fps: List,
        n_examples: int = 10,
        include_similarity_in_prompt: bool = False,
    ) -> Tuple[str, str]:
        """
        Create ICL prompt with similarity-based example selection.
        
        Args:
            query_smiles: Query molecule SMILES
            train_smiles: List of training SMILES
            train_labels: Training labels
            train_fps: Pre-computed fingerprints
            n_examples: Number of examples to select
            include_similarity_in_prompt: Whether to show similarity values in prompt
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Select similar examples
        examples_with_sim = self.select_similar_examples(
            query_smiles, train_smiles, train_labels, train_fps, n_examples
        )
        
        # Temporarily set similarity inclusion
        original_setting = self.config.include_similarity
        self.config.include_similarity = include_similarity_in_prompt
        
        # Format prompt
        query = self.config.query_format.format(smiles=query_smiles)
        examples_str = self.format_examples(examples_with_sim)
        user_content = f"{self.config.task_instruction}\n\n{examples_str}\n\n{query}"
        
        # Restore setting
        self.config.include_similarity = original_setting
        
        return self.config.system_prompt, user_content
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        return (
            f"ICLPromptBuilder(task={self.config.task_name!r}, "
            f"property={self.config.property_name!r})"
        )


# =============================================================================
# Backward Compatibility
# =============================================================================

def create_solubility_prompt(smiles: str, examples: Optional[str] = None) -> Tuple[str, str]:
    """
    Create a structured prompt for solubility prediction.
    
    DEPRECATED: Use ICLPromptBuilder.for_solubility() instead.
    
    Args:
        smiles: SMILES string of the molecule
        examples: Optional few-shot examples string for ICL (pre-formatted)
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    builder = ICLPromptBuilder.for_solubility()
    
    if examples is not None:
        # Legacy format: examples is a pre-formatted string
        query = builder.config.query_format.format(smiles=smiles)
        user_content = f"{builder.config.task_instruction}\n\n{examples}\n\n{query}"
        return builder.config.system_prompt, user_content
    else:
        return builder.create_prompt(smiles, examples=None)