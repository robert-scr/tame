"""Prompt building utilities for molecular property prediction.

This module provides:

**ICL (In-Context Learning) Prompts:**
- `ICLPromptBuilder`: Few-shot ICL prompt builder with factory methods
- `ICLPromptConfig`: Configuration for ICL prompts
- `create_solubility_prompt`: Legacy function for backward compatibility

**CoT (Chain-of-Thought) Generation:**
- `CoTGenerator`: Chain-of-Thought text generation with caching
- `CoTConfig`: Configuration for CoT generation

Both builders support task-specific configurations via factory methods:
- `.for_solubility()` - ESOL/aqueous solubility (log S)
- `.for_lipophilicity()` - Lipophilicity (LogP/LogD)
- `.for_quantum()` - Quantum properties (HOMO, LUMO, gap)
- `.for_generic()` - Custom property prediction

Example:
    >>> from prompts import ICLPromptBuilder, CoTGenerator
    >>> 
    >>> # Create ICL builder for solubility
    >>> icl = ICLPromptBuilder.for_solubility()
    >>> system, user = icl.create_prompt("CCO", examples=[("CCCC", -2.1)])
    >>> 
    >>> # Create CoT generator for solubility
    >>> cot = CoTGenerator.for_solubility(cache_dir="./cache")
    >>> texts = cot.get_cot_texts(smiles_list, client)
"""

from prompts.prompt_builder import (
    ICLPromptBuilder,
    ICLPromptConfig,
    create_solubility_prompt,
    # Pre-defined ICL templates
    SOLUBILITY_ICL_SYSTEM,
    LIPOPHILICITY_ICL_SYSTEM,
    QUANTUM_ICL_SYSTEM,
    GENERIC_ICL_SYSTEM,
)
from prompts.cot_generator import (
    CoTGenerator,
    CoTConfig,
    # Pre-defined CoT templates
    SOLUBILITY_SYSTEM_PROMPT,
    SOLUBILITY_USER_TEMPLATE,
    LIPOPHILICITY_SYSTEM_PROMPT,
    LIPOPHILICITY_USER_TEMPLATE,
    QUANTUM_SYSTEM_PROMPT,
    QUANTUM_USER_TEMPLATE,
    GENERIC_SYSTEM_PROMPT,
    GENERIC_USER_TEMPLATE,
)

__all__ = [
    # ICL prompt builder
    "ICLPromptBuilder",
    "ICLPromptConfig",
    "create_solubility_prompt",  # backward compatibility
    # ICL templates
    "SOLUBILITY_ICL_SYSTEM",
    "LIPOPHILICITY_ICL_SYSTEM",
    "QUANTUM_ICL_SYSTEM",
    "GENERIC_ICL_SYSTEM",
    # CoT generator
    "CoTGenerator",
    "CoTConfig",
    # CoT templates
    "SOLUBILITY_SYSTEM_PROMPT",
    "SOLUBILITY_USER_TEMPLATE",
    "LIPOPHILICITY_SYSTEM_PROMPT",
    "LIPOPHILICITY_USER_TEMPLATE",
    "QUANTUM_SYSTEM_PROMPT",
    "QUANTUM_USER_TEMPLATE",
    "GENERIC_SYSTEM_PROMPT",
    "GENERIC_USER_TEMPLATE",
]
