#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anaphoric ambiguity resolution.

Anaphoric ambiguities occur when pronouns (it, its, they, them, their, this,
that, these, those) have unclear antecedents.
See paper §Methodology (Anaphoric instantiation in Fig. 2).
"""

from common import BaseAmbiguityResolver, run_cli


class AnaphoricResolver(BaseAmbiguityResolver):
    ambiguity_type = "anaphoric"

    ambiguity_definition = (
        "Pronouns (it, its, they, them, their, this, that, these, those) have "
        "unclear antecedents - they could refer to multiple candidate noun "
        "phrases in the requirement."
    )

    cot_steps = [
        "Scan the requirement for pronouns.",
        "For each pronoun, enumerate all candidate antecedents.",
        "Check whether the intended antecedent is clear from context.",
        "If ambiguous, replace the pronoun with the specific antecedent "
        "(use \"[antecedent]'s\" for possessives).",
        "Produce the complete fixed requirement.",
    ]

    critical_rules = [
        "Assume pronouns are ambiguous and need to be replaced.",
        "Replace a pronoun only when its antecedent is unclear or could refer "
        "to multiple candidates.",
        "Use \"[antecedent]'s\" for possessive pronouns (its, their).",
        "Always output the COMPLETE requirement text.",
        "If no pronouns are found, output the original text unchanged.",
    ]

    default_examples = [
        {
            "original": "CS shall accept data from RWIS database in its native format.",
            "fixed":    "CS shall accept data from RWIS database in the RWIS database's native format.",
        },
        {
            "original": "If the request contains storage parameters, it shall create a configuration record from the parameters.",
            "fixed":    "If the request contains storage parameters, the S&T component shall create a configuration record from the parameters.",
        },
        {
            "original": "The system shall log all transactions and send them to the audit server.",
            "fixed":    "The system shall log all transactions and send the transactions to the audit server.",
        },
    ]


if __name__ == "__main__":
    run_cli(AnaphoricResolver, default_csv="data/anaphoric.csv")
