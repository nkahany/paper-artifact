#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vagueness ambiguity resolution.

Vague requirements use imprecise terminology (e.g., 'quickly', 'sufficient',
'appropriate', 'reasonable') that lacks measurable acceptance criteria.
See paper §Methodology.
"""

from common import BaseAmbiguityResolver, run_cli


class VaguenessResolver(BaseAmbiguityResolver):
    ambiguity_type = "vagueness"

    ambiguity_definition = (
        "Vagueness occurs when a requirement uses imprecise terms that lack "
        "measurable or verifiable criteria. Examples include 'quickly', "
        "'sufficient', 'appropriate', 'reasonable', 'user-friendly', "
        "'as soon as possible', 'efficient', 'minimize/maximize'."
    )

    cot_steps = [
        "Scan the requirement for vague or unquantified terms.",
        "For each vague term, identify what aspect is under-specified "
        "(time, quantity, quality, scope).",
        "Decide whether the context provides enough information for a "
        "precise rewrite without fabricating values.",
        "Replace the vague term with measurable criteria when context allows; "
        "otherwise reformulate to make the missing criterion explicit.",
        "Produce the complete fixed requirement.",
    ]

    critical_rules = [
        "Do NOT fabricate specific numerical values that are not supported "
        "by context.",
        "Prefer conservative rewrites that flag the missing criterion "
        "(e.g., 'within a time threshold to be defined') over inventing data.",
        "Preserve the original intent and scope of the requirement.",
        "Always output the COMPLETE requirement text.",
        "If no vague term is found, output the original text unchanged.",
    ]

    default_examples = [
        {
            "original": "The TCS shall be capable of restoring power in sufficient time to avoid loss of air vehicle control.",
            "fixed":    "The TCS shall be capable of restoring power within 30 seconds to avoid loss of air vehicle control.",
        },
        {
            "original": "The system shall respond quickly to user requests.",
            "fixed":    "The system shall respond to user requests within a maximum response time defined in the performance specification.",
        },
        {
            "original": "The application shall provide an appropriate user interface.",
            "fixed":    "The application shall provide a user interface that conforms to the usability criteria specified in the UX guidelines.",
        },
    ]


if __name__ == "__main__":
    run_cli(VaguenessResolver, default_csv="data/vagueness.csv")
