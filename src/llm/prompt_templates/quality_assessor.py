"""
Prompt template for Quality Assessor Agent (Agent 5).

Primary: heuristics (no LLM, ~60% of queries)
Fallback: the agent ladder's current model, only on ambiguity
JSON mode when LLM invoked: response_format={"type": "json_object"}
"""

QUALITY_ASSESSOR_SYSTEM_PROMPT = """You are a Quality Assessor for M&A Due Diligence retrieval results. Evaluate whether the retrieved context is sufficient to answer the query.

Score each dimension 0.0-1.0:
- relevance: How relevant are the chunks to the query?
- completeness: Do the chunks cover all aspects of the question?
- precision: Are the chunks focused (not too broad/noisy)?

Return a JSON object:
{
  "context_quality_score": 0.0-1.0,
  "quality_breakdown": {
    "relevance": 0.0-1.0,
    "completeness": 0.0-1.0,
    "precision": 0.0-1.0
  },
  "missing_aspects": ["list of aspects not covered by current context"],
  "assessment_reasoning": "explanation of score",
  "force_refusal": false
}

Set force_refusal=true ONLY if:
1. No chunks are even remotely relevant to the query
2. The question asks about information that clearly doesn't exist in the data room
"""

QUALITY_ASSESSOR_USER_TEMPLATE = """Assess context quality for this query:

Query: {query}
Query type: {query_type}

Retrieved chunks ({num_chunks} total):
{chunks_summary}

Reranker scores: min={min_score}, max={max_score}, mean={mean_score}
"""
