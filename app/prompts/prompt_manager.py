"""
Prompt Manager — versioned prompts for code explanation.
Supports A/B testing, rollback, and regression detection.
"""

# ─── Code Explanation Prompts ─────────────────────────────

PROMPTS = {
    "v1": """Answer using only the provided context about the codebase.

Context:
{context}

Question:
{query}

Answer:""",

    "v2": """You are an expert code analyst. Answer the question using ONLY the provided code context.

Rules:
- Only use information from the context
- Cite specific functions, classes, or file paths when relevant
- If the answer is not in the context, say "Information not found in the codebase"
- Keep your answer clear and concise (under 200 words)

Context:
{context}

Question:
{query}

Answer:""",

    "v3": """You are a senior software engineer analyzing a Python codebase.

Answer the question using ONLY the provided code and documentation context.

Rules:
1. Only answer from the provided context — never fabricate code or functionality
2. If the information is not available, respond: "This information was not found in the indexed codebase."
3. Reference specific file paths, function names, and line numbers when available
4. Explain code logic clearly, as if helping a teammate understand the codebase
5. Mention dependencies and relationships between components when relevant
6. Keep answer under 250 words unless explaining complex logic

Context:
{context}

Question:
{query}

Answer:""",
}


# ─── Specialized Prompts ─────────────────────────────────

FUNCTION_EXPLAIN_PROMPT = """Explain this function in detail:

Function: {function_name}
File: {file_path}

Code:
{code}

Provide:
1. Purpose — what does this function do?
2. Parameters — what does each argument mean?
3. Return value — what does it return?
4. Logic — step-by-step explanation of the code
5. Dependencies — what other functions/modules does it use?
6. Edge cases — any potential issues or edge cases?

Answer:"""


DEPENDENCY_PROMPT = """Analyze the dependencies for this code component:

Component: {component_name}
File: {file_path}

Code context:
{context}

Provide:
1. Direct imports used
2. Functions/classes it calls
3. Functions/classes that call it
4. External packages depended on
5. Potential circular dependency risks

Answer:"""


CONTEXT_COMPRESSION_PROMPT = """Extract only the information relevant to answering the question.
Remove any code or text that is not directly related.

Question:
{query}

Context:
{context}

Return only the relevant compressed context:"""


VERIFICATION_PROMPT = """Check if the answer is fully supported by the provided context.

Question:
{query}

Context:
{context}

Answer:
{answer}

Return exactly one word: SUPPORTED or UNSUPPORTED"""


HALLUCINATION_CHECK_PROMPT = """Does the answer contain any claims, code references, or functionality 
descriptions that are NOT present in the provided context?

Answer:
{answer}

Context:
{context}

Return exactly: YES or NO"""


CONFIDENCE_SCORE_PROMPT = """Score how confident you are that this answer correctly addresses the question, 
based on the available context.

Question:
{query}

Context:
{context}

Answer:
{answer}

Return a single number from 0 to 100. Nothing else."""


REFLECTION_PROMPT = """Review this answer for completeness and accuracy.
If the answer is incomplete or could be improved using the available context, 
provide an improved version.

Question:
{query}

Context:
{context}

Current answer:
{answer}

Return the improved answer:"""


RETRIEVAL_CONFIDENCE_PROMPT = """Can the provided context answer the question?
Score how well the context covers the information needed.

Question:
{query}

Context:
{context}

Return a single number from 0 to 100. Nothing else."""


# ─── Active Version ──────────────────────────────────────

ACTIVE_VERSION = "v3"


def get_prompt(version: str = None) -> str:
    """Get a prompt template by version."""
    version = version or ACTIVE_VERSION
    return PROMPTS.get(version, PROMPTS[ACTIVE_VERSION])


def get_all_versions() -> list:
    """Return all available prompt versions."""
    return list(PROMPTS.keys())
