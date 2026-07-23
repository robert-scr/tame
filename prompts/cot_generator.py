"""Chain-of-Thought (CoT) prompt generation and text embedding pipeline.

This module provides:
1. CoT prompt templates for molecular property prediction
2. CoT text generation via LLM
3. Text embedding retrieval via Azure OpenAI
4. Disk caching for both CoT texts and embeddings

Usage:
    >>> from prompts.cot_generator import CoTGenerator, CoTConfig
    >>> 
    >>> # Create generator with default solubility config
    >>> cot_gen = CoTGenerator.for_solubility()
    >>> 
    >>> # Generate CoT texts (with caching)
    >>> cot_texts = cot_gen.get_cot_texts(smiles_list)
    >>> 
    >>> # Get embeddings (with caching)
    >>> embeddings = cot_gen.get_embeddings(smiles_list)
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CoTConfig:
    """Configuration for Chain-of-Thought generation.
    
    Attributes:
        task_name: Name of the prediction task (used in cache filenames)
        system_prompt: System prompt for the LLM
        user_prompt_template: User prompt template with {smiles} placeholder
        cot_model: LLM model for CoT generation (e.g., "gpt-5-chat")
        embedding_model: Model for text embeddings (e.g., "text-embedding-3-large")
        embedding_dim: Dimension of text embeddings
        cache_dir: Directory for caching CoT texts and embeddings
        rate_limit_delay: Delay between API calls (seconds)
        batch_size: Batch size for embedding API calls
        multi_query_mode: If True, use MMF paper's 12-question comprehensive approach
        query_subset: Which subset of queries to use ("full", "solubility", "quantum", etc.)
    """
    task_name: str = "solubility"
    system_prompt: str = ""
    user_prompt_template: str = ""
    cot_model: str = "gpt-5-chat"
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    cache_dir: Optional[Path] = None
    rate_limit_delay: float = 0.0
    batch_size: int = 50
    temperature: Optional[float] = None  # None = don't send (required for reasoning models)
    multi_query_mode: bool = False  # True = MMF paper style (12 questions)
    query_subset: str = "full"  # Which queries to use: "full", "solubility", "quantum", etc.
    
    def __post_init__(self):
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "task_name": self.task_name,
            "cot_model": self.cot_model,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "rate_limit_delay": self.rate_limit_delay,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "multi_query_mode": self.multi_query_mode,
            "query_subset": self.query_subset,
        }


# =============================================================================
# Pre-defined CoT Templates (MMF Paper-style Multi-Question Approach)
# =============================================================================

# System prompt for comprehensive molecular analysis (used across all tasks)
MMF_SYSTEM_PROMPT = """You are an expert chemist with deep knowledge of organic chemistry, 
physical chemistry, and molecular properties. Your task is to provide detailed, 
accurate information about molecules based on their SMILES notation.

Provide factual, concise responses based on established chemistry knowledge.
If the molecule is well-known, include specific property values where available.
If not well-known, provide reasoned estimates based on structural features."""

# The 12 query prompts from the MMF paper (Table 16)
MMF_QUERY_TEMPLATES = {
    "structure": """What is the molecular structure of this organic molecule in SMILES notation "{smiles}"?
Could you describe its atoms, bonds, functional groups, and overall arrangement?""",
    
    "physical_properties": """What are the physical properties of the molecule with SMILES "{smiles}"?
Describe its boiling point, melting point, and density.""",
    
    "solubility": """What is the solubility behavior of the molecule with SMILES "{smiles}"?
In which solvents does it dissolve and which does it not?""",
    
    "reactivity": """What is the chemical reactivity of the molecule with SMILES "{smiles}"?
How does it interact with various reagents?""",
    
    "reactions": """Are there any common reactions that the molecule with SMILES "{smiles}" is known to undergo?
Could you describe them?""",
    
    "optical_electrical": """Does the molecule with SMILES "{smiles}" exhibit any unique optical, electrical, or magnetic properties?""",
    
    "chirality": """Is the molecule with SMILES "{smiles}" chiral?
If yes, how does its chirality influence its behavior or properties?""",
    
    "synthesis": """Is the molecule with SMILES "{smiles}" synthesized industrially or in the laboratory?
If yes, could you explain the process?""",
    
    "applications": """Are there any notable uses or applications for the molecule with SMILES "{smiles}" in medicine, industry, or other fields?""",
    
    "natural_occurrence": """Is the molecule with SMILES "{smiles}" found naturally?
If yes, in what sources is it most commonly found?""",
    
    "safety": """What safety measures should be taken when handling the molecule with SMILES "{smiles}"?""",
    
    "environmental": """Are there any environmental impacts associated with the production, use, or disposal of the molecule with SMILES "{smiles}"?""",
}

# Ordered list of queries for comprehensive analysis
MMF_QUERY_ORDER = [
    "structure",
    "physical_properties", 
    "solubility",
    "reactivity",
    "reactions",
    "optical_electrical",
    "chirality",
    "synthesis",
    "applications",
    "natural_occurrence",
    "safety",
    "environmental",
]

# Task-specific query subsets (to reduce API calls for specific tasks)
TASK_QUERY_SUBSETS = {
    "solubility": ["structure", "physical_properties", "solubility", "reactivity"],
    "lipophilicity": ["structure", "physical_properties", "solubility", "reactivity"],
    "quantum": ["structure", "optical_electrical", "chirality", "reactivity"],
    "toxicity": ["structure", "reactivity", "safety", "environmental"],
    "binding": ["structure", "physical_properties", "reactivity", "applications"],  # For BACE/binding affinity
    "full": MMF_QUERY_ORDER,  # All 12 queries
}

# Legacy simple prompts (kept for backward compatibility)
SOLUBILITY_SYSTEM_PROMPT = MMF_SYSTEM_PROMPT
SOLUBILITY_USER_TEMPLATE = """Analyze this molecule for aqueous solubility prediction.

SMILES: {smiles}

Let's think step by step:
1. What functional groups are present?
2. How do these affect solubility?
3. What is the overall polarity?

Provide your analysis in 3-5 sentences."""

LIPOPHILICITY_SYSTEM_PROMPT = MMF_SYSTEM_PROMPT
LIPOPHILICITY_USER_TEMPLATE = """Analyze this molecule for lipophilicity (LogP) prediction.

SMILES: {smiles}

Let's think step by step:
1. What hydrophobic groups are present?
2. What hydrophilic groups are present?
3. How does the structure affect partition coefficient?

Provide your analysis in 3-5 sentences."""

QUANTUM_SYSTEM_PROMPT = MMF_SYSTEM_PROMPT
QUANTUM_USER_TEMPLATE = """Analyze this molecule for quantum property prediction.

SMILES: {smiles}

Let's think step by step:
1. What is the electronic structure?
2. Are there conjugated systems or aromatic rings?
3. How do substituents affect electron distribution?

Provide your analysis in 3-5 sentences."""

GENERIC_SYSTEM_PROMPT = MMF_SYSTEM_PROMPT
GENERIC_USER_TEMPLATE = """Analyze this molecule.

SMILES: {smiles}

Provide a detailed chemical analysis in 3-5 sentences, focusing on:
1. Key structural features
2. Expected chemical behavior
3. Relevant molecular properties"""

# --- Binding Affinity (BACE) ---
BINDING_SYSTEM_PROMPT = """You are an expert medicinal chemist with deep knowledge of drug-target interactions,
structure-activity relationships (SAR), and enzyme inhibition. Your specialty is analyzing
molecules for their potential as enzyme inhibitors, particularly β-secretase (BACE-1) inhibitors.

Provide factual, concise responses based on established medicinal chemistry knowledge.
Focus on structural features relevant to binding affinity and inhibition."""

BINDING_USER_TEMPLATE = """Analyze this molecule as a potential β-secretase (BACE-1) inhibitor.

SMILES: {smiles}

Let's think step by step:
1. What key pharmacophores are present? (amines, amides, hydroxyethylamines, etc.)
2. What structural features might interact with the BACE-1 active site?
3. How do the functional groups affect binding affinity?

Provide your analysis in 3-5 sentences."""

# --- Toxicity (Tox21) ---
TOXICITY_SYSTEM_PROMPT = """You are an expert toxicologist with deep knowledge of molecular toxicology,
nuclear receptor biology, and adverse outcome pathways. Your specialty is analyzing molecules
for potential toxicity through various mechanisms including:

- Nuclear receptor activation (androgen, estrogen, aryl hydrocarbon receptors)
- Stress response pathway activation (oxidative stress, heat shock, DNA damage)
- Mitochondrial toxicity and membrane disruption
- Metabolic activation to reactive intermediates

Provide factual, mechanistically-informed responses based on established toxicology knowledge.
Focus on structural features that may lead to adverse biological effects."""

TOXICITY_USER_TEMPLATE = """Analyze this molecule for potential toxicity.

SMILES: {smiles}

Let's think step by step:
1. What structural alerts for toxicity are present?
2. Which nuclear receptors might this molecule interact with?
3. What metabolic activation pathways might generate reactive metabolites?

Provide your analysis in 3-5 sentences."""


# =============================================================================
# CoT Generator
# =============================================================================

class CoTGenerator:
    """
    Chain-of-Thought text generator with embedding support.
    
    Handles:
    - CoT prompt construction from templates
    - LLM-based CoT text generation
    - Text embedding retrieval
    - Disk caching for both CoT texts and embeddings
    
    Example:
        >>> from prompts.cot_generator import CoTGenerator
        >>> 
        >>> # Using factory method for common tasks
        >>> cot_gen = CoTGenerator.for_solubility(cache_dir="./cache")
        >>> 
        >>> # Or with custom config
        >>> config = CoTConfig(task_name="custom", ...)
        >>> cot_gen = CoTGenerator(config=config)
        >>> 
        >>> # Generate CoT texts
        >>> texts = cot_gen.get_cot_texts(smiles_list, client=azure_client)
        >>> 
        >>> # Get embeddings
        >>> embeddings = cot_gen.get_embeddings(smiles_list, client=azure_client)
    """
    
    def __init__(
        self,
        config: Optional[CoTConfig] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ) -> None:
        """
        Initialize CoT generator.
        
        Args:
            config: CoTConfig instance (if None, uses defaults)
            system_prompt: Override system prompt (if config provided)
            user_prompt_template: Override user template (if config provided)
        """
        self.config = config or CoTConfig()
        
        # Allow overriding prompts
        if system_prompt:
            self.config.system_prompt = system_prompt
        if user_prompt_template:
            self.config.user_prompt_template = user_prompt_template
        
        # Internal caches (memory)
        self._cot_cache: Dict[str, str] = {}
        self._embedding_cache: Dict[str, torch.Tensor] = {}
    
    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def for_solubility(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create a CoT generator configured for solubility prediction."""
        config = CoTConfig(
            task_name="solubility",
            system_prompt=SOLUBILITY_SYSTEM_PROMPT,
            user_prompt_template=SOLUBILITY_USER_TEMPLATE,
            cache_dir=Path(cache_dir) if cache_dir else None,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_lipophilicity(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create a CoT generator configured for lipophilicity prediction."""
        config = CoTConfig(
            task_name="lipophilicity",
            system_prompt=LIPOPHILICITY_SYSTEM_PROMPT,
            user_prompt_template=LIPOPHILICITY_USER_TEMPLATE,
            cache_dir=Path(cache_dir) if cache_dir else None,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_quantum(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create a CoT generator configured for quantum property prediction."""
        config = CoTConfig(
            task_name="quantum",
            system_prompt=QUANTUM_SYSTEM_PROMPT,
            user_prompt_template=QUANTUM_USER_TEMPLATE,
            cache_dir=Path(cache_dir) if cache_dir else None,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_generic(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create a CoT generator for generic molecular analysis."""
        config = CoTConfig(
            task_name="generic",
            system_prompt=GENERIC_SYSTEM_PROMPT,
            user_prompt_template=GENERIC_USER_TEMPLATE,
            cache_dir=Path(cache_dir) if cache_dir else None,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_mmf(
        cls, 
        task_name: str = "mmf",
        query_subset: str = "full",
        cache_dir: Optional[Union[str, Path]] = None, 
        **kwargs
    ) -> "CoTGenerator":
        """
        Create a CoT generator using MMF paper's comprehensive multi-question approach.
        
        This uses 12 detailed queries to gather comprehensive molecular information:
        - Structure (atoms, bonds, functional groups)
        - Physical properties (bp, mp, density)
        - Solubility behavior
        - Chemical reactivity
        - Common reactions
        - Optical/electrical/magnetic properties
        - Chirality
        - Synthesis methods
        - Applications
        - Natural occurrence
        - Safety measures
        - Environmental impacts
        
        Args:
            task_name: Name for caching (default: "mmf")
            query_subset: Which queries to use:
                - "full": All 12 queries (most comprehensive, expensive)
                - "solubility": Structure, physical, solubility, reactivity
                - "lipophilicity": Same as solubility
                - "quantum": Structure, optical/electrical, chirality, reactivity
                - "toxicity": Structure, reactivity, safety, environmental
            cache_dir: Directory for caching
            **kwargs: Additional config options
            
        Returns:
            CoTGenerator configured for MMF-style comprehensive analysis
        """
        config = CoTConfig(
            task_name=task_name,
            system_prompt=MMF_SYSTEM_PROMPT,
            user_prompt_template="",  # Not used in multi-query mode
            multi_query_mode=True,
            query_subset=query_subset,
            cache_dir=Path(cache_dir) if cache_dir else None,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_solubility_mmf(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create MMF-style generator optimized for solubility prediction (4 queries)."""
        return cls.for_mmf(
            task_name="solubility_mmf",
            query_subset="solubility",
            cache_dir=cache_dir,
            **kwargs
        )
    
    @classmethod
    def for_quantum_mmf(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create MMF-style generator optimized for quantum property prediction (4 queries)."""
        return cls.for_mmf(
            task_name="quantum_mmf",
            query_subset="quantum",
            cache_dir=cache_dir,
            **kwargs
        )
    
    @classmethod
    def for_solubility_fast(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create FAST single-query generator for solubility (1 LLM call per molecule).
        
        This is much faster than MMF-style (4 queries) but less comprehensive.
        Uses a single prompt that covers structure, polarity, and solubility reasoning.
        
        Cache is named 'solubility_fast' to avoid conflicts with MMF cache.
        """
        # Comprehensive single prompt covering key solubility factors
        user_template = """Analyze this molecule for aqueous solubility prediction.

SMILES: {smiles}

Provide a comprehensive analysis covering:
1. Molecular structure: key functional groups, atoms, bonds
2. Physical properties: approximate molecular weight, polarity
3. Solubility factors: hydrophilic/hydrophobic groups, H-bond donors/acceptors
4. Expected solubility behavior in water and organic solvents

Provide your analysis in 4-6 sentences."""

        config = CoTConfig(
            task_name="solubility_fast",  # Different cache name!
            system_prompt=MMF_SYSTEM_PROMPT,
            user_prompt_template=user_template,
            multi_query_mode=False,  # Single query = fast
            cache_dir=Path(cache_dir) if cache_dir else None,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_quantum_fast(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create FAST single-query generator for quantum properties (1 LLM call per molecule).
        
        Cache is named 'quantum_fast' to avoid conflicts with MMF cache.
        """
        user_template = """Analyze this molecule for quantum property prediction.

SMILES: {smiles}

Provide a comprehensive analysis covering:
1. Molecular structure: atoms, bonds, functional groups, geometry
2. Electronic structure: conjugation, aromaticity, electron distribution
3. Frontier orbitals: expected HOMO/LUMO characteristics
4. Optical/electronic properties: expected absorption, emission, conductivity

Provide your analysis in 4-6 sentences."""

        config = CoTConfig(
            task_name="quantum_fast",  # Different cache name!
            system_prompt=MMF_SYSTEM_PROMPT,
            user_prompt_template=user_template,
            multi_query_mode=False,  # Single query = fast
            cache_dir=Path(cache_dir) if cache_dir else None,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_binding(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create a CoT generator configured for binding affinity (BACE) prediction."""
        config = CoTConfig(
            task_name="binding",
            system_prompt=BINDING_SYSTEM_PROMPT,
            user_prompt_template=BINDING_USER_TEMPLATE,
            cache_dir=Path(cache_dir) if cache_dir else None,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_binding_fast(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create FAST single-query generator for binding affinity (1 LLM call per molecule).
        
        Optimized for BACE-1 inhibitor analysis. Covers key pharmacophore features,
        binding site interactions, and structure-activity relationships.
        
        Cache is named 'binding_fast' to avoid conflicts with MMF cache.
        """
        user_template = """Analyze this molecule as a potential β-secretase (BACE-1) inhibitor.

SMILES: {smiles}

Provide a comprehensive analysis covering:
1. Core scaffold: Identify the main structural framework (e.g., aminothiazine, hydroxyethylamine, macrocycle)
2. Key pharmacophores: Amine groups, amide bonds, hydroxyl groups that can form H-bonds
3. Lipophilic groups: Aromatic rings, alkyl chains that fill hydrophobic pockets
4. Drug-like properties: Size, hydrogen bond donors/acceptors, expected cell permeability
5. SAR insights: How substituents might affect binding to the BACE-1 active site aspartates

Provide your analysis in 5-7 sentences."""

        config = CoTConfig(
            task_name="binding_fast",  # Different cache name!
            system_prompt=BINDING_SYSTEM_PROMPT,
            user_prompt_template=user_template,
            multi_query_mode=False,  # Single query = fast
            cache_dir=Path(cache_dir) if cache_dir else None,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_binding_mmf(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create MMF-style generator optimized for binding affinity prediction (4 queries)."""
        return cls.for_mmf(
            task_name="binding_mmf",
            query_subset="binding",
            cache_dir=cache_dir,
            **kwargs
        )
    
    @classmethod
    def for_toxicity_fast(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create FAST single-query generator for toxicity prediction (1 LLM call per molecule).
        
        Optimized for Tox21-style toxicity prediction. Covers nuclear receptor activation,
        stress response pathways, and general toxicity mechanisms.
        
        Cache is named 'toxicity_fast' to avoid conflicts with MMF cache.
        """
        user_template = """Analyze this molecule for toxicity prediction across multiple biological endpoints.

SMILES: {smiles}

Provide a comprehensive toxicity analysis covering:
1. Structural alerts: Known toxic substructures (aromatic amines, nitro groups, epoxides, Michael acceptors)
2. Receptor binding potential: Nuclear receptor interactions (AR, ER, AhR, PPAR-gamma)
3. Stress response: Potential for oxidative stress, mitochondrial toxicity, DNA damage
4. Metabolic activation: Likely CYP450 metabolism and reactive metabolite formation
5. Physical properties: Lipophilicity and membrane permeability affecting bioaccumulation

Provide your analysis in 5-7 sentences."""

        config = CoTConfig(
            task_name="toxicity_fast",
            system_prompt=TOXICITY_SYSTEM_PROMPT,
            user_prompt_template=user_template,
            multi_query_mode=False,  # Single query = fast
            cache_dir=Path(cache_dir) if cache_dir else None,
            **kwargs
        )
        return cls(config=config)
    
    @classmethod
    def for_toxicity_mmf(cls, cache_dir: Optional[Union[str, Path]] = None, **kwargs) -> "CoTGenerator":
        """Create MMF-style generator optimized for toxicity prediction (4 queries)."""
        return cls.for_mmf(
            task_name="toxicity_mmf",
            query_subset="toxicity",
            cache_dir=cache_dir,
            **kwargs
        )

    # -------------------------------------------------------------------------
    # Prompt Construction
    # -------------------------------------------------------------------------
    
    def create_prompt(self, smiles: str) -> Tuple[str, str]:
        """
        Create CoT prompt for a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        user_prompt = self.config.user_prompt_template.format(smiles=smiles)
        return self.config.system_prompt, user_prompt
    
    def get_query_list(self) -> List[str]:
        """
        Get the list of query keys to use based on config.
        
        Returns:
            List of query keys (e.g., ["structure", "solubility", ...])
        """
        subset = self.config.query_subset
        if subset in TASK_QUERY_SUBSETS:
            return TASK_QUERY_SUBSETS[subset]
        return MMF_QUERY_ORDER  # Default to full
    
    def create_mmf_prompts(self, smiles: str) -> List[Tuple[str, str]]:
        """
        Create MMF-style multi-question prompts for a molecule.
        
        This follows the paper's approach of asking 12 comprehensive questions
        about each molecule to gather rich semantic information.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of (system_prompt, user_prompt) tuples for each query
        """
        query_keys = self.get_query_list()
        prompts = []
        
        for key in query_keys:
            if key in MMF_QUERY_TEMPLATES:
                user_prompt = MMF_QUERY_TEMPLATES[key].format(smiles=smiles)
                prompts.append((MMF_SYSTEM_PROMPT, user_prompt))
        
        return prompts
    
    # -------------------------------------------------------------------------
    # CoT Text Generation
    # -------------------------------------------------------------------------
    
    def generate_cot_text(self, smiles: str, client) -> str:
        """
        Generate CoT reasoning text for a single molecule.
        
        If multi_query_mode is enabled, generates comprehensive text using
        all configured queries (MMF paper style). Otherwise uses single prompt.
        
        Args:
            smiles: SMILES string
            client: Azure OpenAI client
            
        Returns:
            Generated CoT text (concatenated if multi-query mode)
        """
        if self.config.multi_query_mode:
            return self.generate_mmf_cot_text(smiles, client)
        
        # Single prompt mode (legacy)
        system_prompt, user_prompt = self.create_prompt(smiles)
        
        kwargs: Dict[str, Any] = dict(
            model=self.config.cot_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    
    def generate_mmf_cot_text(self, smiles: str, client) -> str:
        """
        Generate comprehensive CoT text using MMF paper's multi-question approach.
        
        Asks multiple questions about the molecule and concatenates all responses
        into a single comprehensive description.
        
        Args:
            smiles: SMILES string
            client: Azure OpenAI client
            
        Returns:
            Concatenated comprehensive molecular description
        """
        prompts = self.create_mmf_prompts(smiles)
        query_keys = self.get_query_list()
        
        all_responses = []
        
        for i, (system_prompt, user_prompt) in enumerate(prompts):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self.config.cot_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                if self.config.temperature is not None:
                    kwargs["temperature"] = self.config.temperature
                
                response = client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content.strip()
                
                # Add section header for clarity
                section_name = query_keys[i].replace("_", " ").title()
                all_responses.append(f"[{section_name}]\n{text}")
                
                # Rate limiting between queries
                if self.config.rate_limit_delay > 0:
                    time.sleep(self.config.rate_limit_delay)
                    
            except Exception as e:
                all_responses.append(f"[{query_keys[i]}]\nError: {str(e)}")
        
        # Concatenate all responses with separators
        return "\n\n".join(all_responses)
    
    def generate_mmf_cot_dict(self, smiles: str, client) -> Dict[str, str]:
        """
        Generate comprehensive CoT text and return as dictionary.
        
        Useful when you want to access individual query responses separately.
        
        Args:
            smiles: SMILES string
            client: Azure OpenAI client
            
        Returns:
            Dict mapping query_key -> response text
        """
        prompts = self.create_mmf_prompts(smiles)
        query_keys = self.get_query_list()
        
        results = {}
        
        for i, (system_prompt, user_prompt) in enumerate(prompts):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self.config.cot_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                if self.config.temperature is not None:
                    kwargs["temperature"] = self.config.temperature
                
                response = client.chat.completions.create(**kwargs)
                results[query_keys[i]] = response.choices[0].message.content.strip()
                
                if self.config.rate_limit_delay > 0:
                    time.sleep(self.config.rate_limit_delay)
                    
            except Exception as e:
                results[query_keys[i]] = f"Error: {str(e)}"
        
        return results
    
    def get_cot_texts(
        self,
        smiles_list: List[str],
        client,
        verbose: bool = True,
    ) -> Dict[str, str]:
        """
        Get CoT texts for a list of molecules (with caching).
        
        Args:
            smiles_list: List of SMILES strings
            client: Azure OpenAI client
            verbose: Show progress bar
            
        Returns:
            Dict mapping SMILES -> CoT text
        """
        # Load from disk cache if available
        self._load_cot_cache()
        
        # Find missing SMILES
        missing = [s for s in smiles_list if s not in self._cot_cache]
        
        if verbose:
            cached_count = len(smiles_list) - len(missing)
            print(f"CoT texts: {cached_count} cached, {len(missing)} to generate")
        
        # Generate missing texts
        if missing:
            iterator = tqdm(missing, desc="Generating CoT texts") if verbose else missing
            
            for i, smi in enumerate(iterator):
                try:
                    cot_text = self.generate_cot_text(smi, client)
                    self._cot_cache[smi] = cot_text
                except Exception as e:
                    if verbose:
                        print(f"\nWarning: Failed for {smi}: {e}")
                    # Fallback: simple description
                    self._cot_cache[smi] = f"Molecule with SMILES: {smi}"
                
                # Rate limiting
                if (i + 1) % 10 == 0:
                    time.sleep(self.config.rate_limit_delay * 10)
            
            # Save updated cache
            self._save_cot_cache()
        
        # Return only requested SMILES
        return {s: self._cot_cache[s] for s in smiles_list if s in self._cot_cache}

    def get_cot_texts_parallel(
        self,
        smiles_list: List[str],
        client,
        max_workers: int = 8,
        verbose: bool = True,
        save_interval: int = 100,
    ) -> Dict[str, str]:
        """
        Get CoT texts for a list of molecules using parallel execution.
        
        Uses ThreadPoolExecutor for concurrent API calls. Significantly faster
        than sequential processing (~8x speedup with max_workers=8).
        
        Args:
            smiles_list: List of SMILES strings
            client: Azure OpenAI client
            max_workers: Number of parallel threads (default: 8)
            verbose: Show progress bar
            save_interval: Save cache every N molecules
            
        Returns:
            Dict mapping SMILES -> CoT text
        """
        # Load from disk cache if available
        self._load_cot_cache()
        
        # Find missing SMILES
        missing = [s for s in smiles_list if s not in self._cot_cache]
        
        if verbose:
            cached_count = len(smiles_list) - len(missing)
            print(f"CoT texts: {cached_count} cached, {len(missing)} to generate ({max_workers} workers)")
        
        if not missing:
            return {s: self._cot_cache[s] for s in smiles_list if s in self._cot_cache}
        
        # Worker function for a single SMILES
        _first_error_logged = False
        def process_smiles(smi: str) -> Tuple[str, str]:
            nonlocal _first_error_logged
            try:
                cot_text = self.generate_cot_text(smi, client)
                return smi, cot_text
            except Exception as e:
                if not _first_error_logged:
                    _first_error_logged = True
                    log.warning(f"CoT generation failed (showing first error only): {type(e).__name__}: {e}")
                return smi, f"Molecule with SMILES: {smi}"
        
        # Process in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_smiles, smi): smi for smi in missing}
            
            pbar = tqdm(total=len(missing), desc="Generating CoT texts (parallel)") if verbose else None
            
            for future in as_completed(futures):
                smi, cot_text = future.result()
                self._cot_cache[smi] = cot_text
                completed += 1
                
                if pbar:
                    pbar.update(1)
                
                # Periodic cache save
                if completed % save_interval == 0:
                    self._save_cot_cache()
            
            if pbar:
                pbar.close()
        
        # Final save
        self._save_cot_cache()
        
        if verbose:
            print(f"✓ Generated {len(missing)} CoT texts")
        
        return {s: self._cot_cache[s] for s in smiles_list if s in self._cot_cache}
    
    # -------------------------------------------------------------------------
    # Text Embeddings
    # -------------------------------------------------------------------------
    
    def get_text_embeddings_batch(self, texts: List[str], client) -> torch.Tensor:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            client: Azure OpenAI client
            
        Returns:
            Tensor of shape (len(texts), embedding_dim)
        """
        response = client.embeddings.create(
            input=texts,
            model=self.config.embedding_model,
        )
        embeddings = [item.embedding for item in response.data]
        return torch.tensor(embeddings)
    
    def get_embeddings(
        self,
        smiles_list: List[str],
        client,
        cot_texts: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Get text embeddings for a list of molecules (with caching).
        
        If cot_texts is not provided, will generate them first.
        
        Args:
            smiles_list: List of SMILES strings
            client: Azure OpenAI client
            cot_texts: Pre-computed CoT texts (optional)
            verbose: Show progress bar
            
        Returns:
            Tensor of shape (len(smiles_list), embedding_dim)
        """
        # Load from disk cache if available
        self._load_embedding_cache()
        
        # Get CoT texts if not provided
        if cot_texts is None:
            cot_texts = self.get_cot_texts(smiles_list, client, verbose=verbose)
        
        # Find missing embeddings
        missing = [s for s in smiles_list if s not in self._embedding_cache]
        
        if verbose:
            cached_count = len(smiles_list) - len(missing)
            print(f"Embeddings: {cached_count} cached, {len(missing)} to compute")
        
        # Compute missing embeddings in batches
        if missing:
            batch_size = self.config.batch_size
            
            for start in tqdm(
                range(0, len(missing), batch_size),
                desc="Computing embeddings",
                disable=not verbose
            ):
                batch_smiles = missing[start:start + batch_size]
                batch_texts = [cot_texts.get(s, f"Molecule: {s}") for s in batch_smiles]
                
                try:
                    batch_emb = self.get_text_embeddings_batch(batch_texts, client)
                    
                    for i, smi in enumerate(batch_smiles):
                        self._embedding_cache[smi] = batch_emb[i]
                        
                except Exception as e:
                    if verbose:
                        print(f"\nWarning: Batch failed: {e}")
                    # Fall back to individual requests
                    for smi in batch_smiles:
                        try:
                            emb = self.get_text_embeddings_batch(
                                [cot_texts.get(smi, f"Molecule: {smi}")], client
                            )
                            self._embedding_cache[smi] = emb[0]
                        except:
                            self._embedding_cache[smi] = torch.zeros(self.config.embedding_dim)
                
                time.sleep(self.config.rate_limit_delay)
            
            # Save updated cache
            self._save_embedding_cache()
        
        # Stack embeddings in order
        embeddings = []
        for smi in smiles_list:
            if smi in self._embedding_cache:
                emb = self._embedding_cache[smi]
                if isinstance(emb, torch.Tensor):
                    embeddings.append(emb)
                else:
                    embeddings.append(torch.tensor(emb))
            else:
                embeddings.append(torch.zeros(self.config.embedding_dim))
        
        return torch.stack(embeddings)

    def get_embeddings_parallel(
        self,
        smiles_list: List[str],
        client,
        cot_texts: Optional[Dict[str, str]] = None,
        max_workers: int = 8,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Get text embeddings for a list of molecules using parallel execution.
        
        Uses ThreadPoolExecutor for concurrent CoT generation and batch embedding.
        Significantly faster than sequential processing.
        
        Args:
            smiles_list: List of SMILES strings
            client: Azure OpenAI client
            cot_texts: Pre-computed CoT texts (optional)
            max_workers: Number of parallel threads for CoT generation
            verbose: Show progress bar
            
        Returns:
            Tensor of shape (len(smiles_list), embedding_dim)
        """
        # Load from disk cache if available
        self._load_embedding_cache()
        
        # Get CoT texts if not provided (using parallel method)
        if cot_texts is None:
            cot_texts = self.get_cot_texts_parallel(
                smiles_list, client, max_workers=max_workers, verbose=verbose
            )
        
        # Find missing embeddings
        missing = [s for s in smiles_list if s not in self._embedding_cache]
        
        if verbose:
            cached_count = len(smiles_list) - len(missing)
            print(f"Embeddings: {cached_count} cached, {len(missing)} to compute")
        
        # Compute missing embeddings in batches (embeddings API is already batched)
        if missing:
            batch_size = self.config.batch_size
            
            for start in tqdm(
                range(0, len(missing), batch_size),
                desc="Computing embeddings",
                disable=not verbose
            ):
                batch_smiles = missing[start:start + batch_size]
                batch_texts = [cot_texts.get(s, f"Molecule: {s}") for s in batch_smiles]
                
                try:
                    batch_emb = self.get_text_embeddings_batch(batch_texts, client)
                    
                    for i, smi in enumerate(batch_smiles):
                        self._embedding_cache[smi] = batch_emb[i]
                        
                except Exception as e:
                    if verbose:
                        print(f"\nWarning: Batch failed: {e}")
                    # Fall back to individual requests
                    for smi in batch_smiles:
                        try:
                            emb = self.get_text_embeddings_batch(
                                [cot_texts.get(smi, f"Molecule: {smi}")], client
                            )
                            self._embedding_cache[smi] = emb[0]
                        except:
                            self._embedding_cache[smi] = torch.zeros(self.config.embedding_dim)
                
                time.sleep(self.config.rate_limit_delay)
            
            # Save updated cache
            self._save_embedding_cache()
        
        # Stack embeddings in order
        embeddings = []
        for smi in smiles_list:
            if smi in self._embedding_cache:
                emb = self._embedding_cache[smi]
                if isinstance(emb, torch.Tensor):
                    embeddings.append(emb)
                else:
                    embeddings.append(torch.tensor(emb))
            else:
                embeddings.append(torch.zeros(self.config.embedding_dim))
        
        return torch.stack(embeddings)
    
    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------
    
    @property
    def _cot_cache_path(self) -> Optional[Path]:
        """Path to CoT text cache file."""
        if self.config.cache_dir is None:
            return None
        return self.config.cache_dir / f"{self.config.task_name}_cot_texts.json"
    
    @property
    def _embedding_cache_path(self) -> Optional[Path]:
        """Path to embedding cache file."""
        if self.config.cache_dir is None:
            return None
        return self.config.cache_dir / f"{self.config.task_name}_text_embeddings.pkl"
    
    def _load_cot_cache(self) -> None:
        """Load CoT cache from disk."""
        if self._cot_cache_path and self._cot_cache_path.exists():
            with open(self._cot_cache_path, "r", encoding="utf-8") as f:
                disk_cache = json.load(f)
            # Merge with memory cache (memory takes precedence)
            for k, v in disk_cache.items():
                if k not in self._cot_cache:
                    self._cot_cache[k] = v
    
    def _save_cot_cache(self) -> None:
        """Save CoT cache to disk."""
        if self._cot_cache_path:
            self._cot_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cot_cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cot_cache, f, indent=2, ensure_ascii=False)
    
    def _load_embedding_cache(self) -> None:
        """Load embedding cache from disk."""
        if self._embedding_cache_path and self._embedding_cache_path.exists():
            with open(self._embedding_cache_path, "rb") as f:
                disk_cache = pickle.load(f)
            # Merge with memory cache
            for k, v in disk_cache.items():
                if k not in self._embedding_cache:
                    self._embedding_cache[k] = v
    
    def _save_embedding_cache(self) -> None:
        """Save embedding cache to disk."""
        if self._embedding_cache_path:
            self._embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._embedding_cache_path, "wb") as f:
                pickle.dump(self._embedding_cache, f)
    
    def clear_cache(self, cot: bool = True, embeddings: bool = True) -> None:
        """
        Clear caches (memory and optionally disk).
        
        Args:
            cot: Clear CoT text cache
            embeddings: Clear embedding cache
        """
        if cot:
            self._cot_cache = {}
            if self._cot_cache_path and self._cot_cache_path.exists():
                self._cot_cache_path.unlink()
        
        if embeddings:
            self._embedding_cache = {}
            if self._embedding_cache_path and self._embedding_cache_path.exists():
                self._embedding_cache_path.unlink()
    
    def load_cot_texts_from_file(self, filepath: Union[str, Path]) -> Dict[str, str]:
        """
        Load CoT texts from a custom file path.
        
        Args:
            filepath: Path to JSON file with {smiles: cot_text} mapping
            
        Returns:
            Dict of loaded CoT texts
        """
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        # Merge into cache
        self._cot_cache.update(loaded)
        return loaded
    
    def load_embeddings_from_file(self, filepath: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Load embeddings from a custom file path.
        
        Args:
            filepath: Path to pickle file with {smiles: tensor} mapping
            
        Returns:
            Dict of loaded embeddings
        """
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            loaded = pickle.load(f)
        
        # Merge into cache
        self._embedding_cache.update(loaded)
        return loaded
    
    def save_cot_texts_to_file(self, filepath: Union[str, Path]) -> None:
        """Save current CoT cache to a custom file path."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._cot_cache, f, indent=2, ensure_ascii=False)
    
    def save_embeddings_to_file(self, filepath: Union[str, Path]) -> None:
        """Save current embedding cache to a custom file path."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self._embedding_cache, f)
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data (memory and disk)."""
        stats = {
            "cot_texts_in_memory": len(self._cot_cache),
            "embeddings_in_memory": len(self._embedding_cache),
            "cot_cache_path": str(self._cot_cache_path) if self._cot_cache_path else None,
            "embedding_cache_path": str(self._embedding_cache_path) if self._embedding_cache_path else None,
        }
        
        # Check disk cache sizes
        if self._cot_cache_path and self._cot_cache_path.exists():
            with open(self._cot_cache_path, "r", encoding="utf-8") as f:
                disk_cot = json.load(f)
            stats["cot_texts_on_disk"] = len(disk_cot)
        else:
            stats["cot_texts_on_disk"] = 0
            
        if self._embedding_cache_path and self._embedding_cache_path.exists():
            with open(self._embedding_cache_path, "rb") as f:
                disk_emb = pickle.load(f)
            stats["embeddings_on_disk"] = len(disk_emb)
        else:
            stats["embeddings_on_disk"] = 0
        
        return stats
    
    def has_cot_texts(self, smiles_list: List[str]) -> Dict[str, bool]:
        """
        Check which SMILES have cached CoT texts.
        
        Useful for determining if expensive LLM calls are needed.
        
        Args:
            smiles_list: List of SMILES to check
            
        Returns:
            Dict mapping SMILES -> bool (True if cached)
        """
        self._load_cot_cache()
        return {s: s in self._cot_cache for s in smiles_list}
    
    def has_embeddings(self, smiles_list: List[str]) -> Dict[str, bool]:
        """
        Check which SMILES have cached embeddings.
        
        Args:
            smiles_list: List of SMILES to check
            
        Returns:
            Dict mapping SMILES -> bool (True if cached)
        """
        self._load_embedding_cache()
        return {s: s in self._embedding_cache for s in smiles_list}
    
    def get_missing_cot_texts(self, smiles_list: List[str]) -> List[str]:
        """
        Get list of SMILES that don't have cached CoT texts.
        
        Useful for estimating API costs before generation.
        
        Args:
            smiles_list: List of SMILES to check
            
        Returns:
            List of SMILES without cached CoT texts
        """
        self._load_cot_cache()
        return [s for s in smiles_list if s not in self._cot_cache]
    
    def get_missing_embeddings(self, smiles_list: List[str]) -> List[str]:
        """
        Get list of SMILES that don't have cached embeddings.
        
        Args:
            smiles_list: List of SMILES to check
            
        Returns:
            List of SMILES without cached embeddings
        """
        self._load_embedding_cache()
        return [s for s in smiles_list if s not in self._embedding_cache]
    
    def estimate_cost(
        self, 
        smiles_list: List[str],
        cot_cost_per_call: float = 0.01,  # Approximate cost per LLM call
        embedding_cost_per_1k_tokens: float = 0.0001,  # Embedding API cost
        avg_cot_tokens: int = 200,  # Average CoT text length in tokens (per query)
    ) -> Dict[str, Any]:
        """
        Estimate API costs for generating CoT texts and embeddings.
        
        This helps plan expensive operations by showing what's cached vs needed.
        Accounts for multi-query mode where each molecule requires multiple LLM calls.
        
        Args:
            smiles_list: List of SMILES to process
            cot_cost_per_call: Estimated cost per LLM API call (USD)
            embedding_cost_per_1k_tokens: Embedding cost per 1K tokens (USD)
            avg_cot_tokens: Average tokens per CoT text (per query in multi-query mode)
            
        Returns:
            Cost estimation dict
        """
        missing_cot = self.get_missing_cot_texts(smiles_list)
        missing_emb = self.get_missing_embeddings(smiles_list)
        
        # Calculate number of queries per molecule
        if self.config.multi_query_mode:
            queries_per_molecule = len(self.get_query_list())
            total_tokens_per_molecule = avg_cot_tokens * queries_per_molecule
        else:
            queries_per_molecule = 1
            total_tokens_per_molecule = avg_cot_tokens
        
        # Cost calculations
        llm_calls_needed = len(missing_cot) * queries_per_molecule
        cot_cost = llm_calls_needed * cot_cost_per_call
        emb_cost = len(missing_emb) * (total_tokens_per_molecule / 1000) * embedding_cost_per_1k_tokens
        
        result = {
            "total_molecules": len(smiles_list),
            "cot_texts_cached": len(smiles_list) - len(missing_cot),
            "cot_texts_needed": len(missing_cot),
            "embeddings_cached": len(smiles_list) - len(missing_emb),
            "embeddings_needed": len(missing_emb),
            "estimated_cot_cost_usd": cot_cost,
            "estimated_embedding_cost_usd": emb_cost,
            "estimated_total_cost_usd": cot_cost + emb_cost,
        }
        
        if self.config.multi_query_mode:
            result["multi_query_mode"] = True
            result["queries_per_molecule"] = queries_per_molecule
            result["total_llm_calls_needed"] = llm_calls_needed
            result["query_subset"] = self.config.query_subset
            result["note"] = f"MMF mode: {queries_per_molecule} queries/molecule. Consider using a subset to reduce costs."
        else:
            result["multi_query_mode"] = False
            result["note"] = "Single-query mode. Use for_mmf() for comprehensive analysis."
        
        return result
    
    def __repr__(self) -> str:
        if self.config.multi_query_mode:
            return (
                f"CoTGenerator(task={self.config.task_name!r}, "
                f"mode='mmf-{self.config.query_subset}' ({len(self.get_query_list())} queries), "
                f"cot_model={self.config.cot_model!r})"
            )
        return (
            f"CoTGenerator(task={self.config.task_name!r}, "
            f"mode='single-query', "
            f"cot_model={self.config.cot_model!r})"
        )
