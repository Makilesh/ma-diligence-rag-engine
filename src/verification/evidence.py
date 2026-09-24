"""
The evidence text of a retrieved chunk — shared by synthesis and verification.

Retrieval returns small child chunks for precision and attaches each child's
~2,048-token parent section as `parent_text` for context. The child is a
substring of its parent, so "child + parent" repeats the child, and several
children of one parent repeat the whole parent. Both the synthesizer and the
validator read evidence through here so they see the same text: a validator
shown less than the synthesizer flags grounded claims as hallucinations (this
happened when the validator truncated chunks to 500 characters).
"""

from __future__ import annotations


def chunk_evidence_text(chunk: dict) -> str:
    """
    Full evidence text for one chunk: its parent section when the parent
    contains it, otherwise the chunk followed by its parent context.

    Args:
        chunk: Retrieved chunk payload.

    Returns:
        Evidence text (may be empty).
    """
    text = chunk.get("text", "") or ""
    parent = chunk.get("parent_text", "") or ""
    if not parent:
        return text
    if text.strip() and text.strip() in parent:
        return parent
    return f"{text}\n[Parent context]: {parent}" if text else parent


def parent_key(chunk: dict) -> str | None:
    """
    Identity of the parent section a chunk expands to, if it has one.

    Args:
        chunk: Retrieved chunk payload.

    Returns:
        parent_chunk_id when present, else None.
    """
    pid = chunk.get("parent_chunk_id")
    return str(pid) if pid and chunk.get("parent_text") else None
