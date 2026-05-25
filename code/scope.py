#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scope ambiguity resolution.

Scope ambiguities arise from unclear boundaries of quantifiers, negation,
modifiers, or conjunctions (e.g., 'every ... not ...', 'all ... and ...').
See paper §Methodology and Kamath et al. (2024).
"""

from common import BaseAmbiguityResolver, run_cli


class ScopeResolver(BaseAmbiguityResolver):
    ambiguity_type = "scope"

    ambiguity_definition = (
        "Scope ambiguity occurs when the boundaries of quantifiers (each, all, "
        "every, any, some), negation, or operators are unclear. Common types "
        "include quantifier-quantifier, quantifier-negation, and "
        "quantifier-adverb interactions, which admit both surface and inverse "
        "scope readings."
    )

    cot_steps = [
        "Identify all quantifiers, negations, and conjunctions in the "
        "requirement.",
        "Enumerate the possible scope readings (surface vs. inverse).",
        "Determine which reading is intended based on context or domain "
        "knowledge.",
        "Rewrite to make the intended scope explicit (e.g., distributive vs. "
        "collective, universal vs. existential).",
        "Produce the complete fixed requirement.",
    ]

    critical_rules = [
        "Make quantifier scope explicit by rephrasing rather than relying on "
        "word order alone.",
        "When the requirement involves negation, place it unambiguously "
        "relative to the quantifier.",
        "Preserve the original universal/existential intent; do NOT silently "
        "switch a universal reading to a distributive one (or vice versa).",
        "Always output the COMPLETE requirement text.",
        "If scope is already clear, output the original text unchanged.",
    ]

    default_examples = [
        {
            "original": "The system shall automatically update all software components when any new version is available.",
            "fixed":    "The system shall automatically update each software component when a new version of that specific component becomes available.",
        },
        {
            "original": "The EMC shall be able to maintain an equipment list for each station.",
            "fixed":    "The EMC shall maintain, for every station, a dedicated equipment list specific to that station.",
        },
        {
            "original": "Each transaction must be approved by a manager.",
            "fixed":    "For every transaction, there must exist at least one manager who approves that transaction (the approving manager may differ between transactions).",
        },
    ]


if __name__ == "__main__":
    run_cli(ScopeResolver, default_csv="data/scope.csv")
