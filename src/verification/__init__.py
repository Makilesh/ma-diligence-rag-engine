"""
Deterministic and local-model answer verification.

The pieces here replace an LLM-judges-LLM validator with checks that are
reproducible and free to run:

- numeric_grounding: every figure in an answer traced to a figure in the
  retrieved context, after unit normalisation.
- claims: splitting an answer into atomic, checkable claims.
- nli: a local cross-encoder NLI model scoring claim-vs-evidence entailment.
- claim_checker: combines the above, calling an LLM judge only for the claims
  neither deterministic check could decide.
- prompt_safety: delimiting untrusted document text inside prompts.
"""
