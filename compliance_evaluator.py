"""
compliance_evaluator.py
Mock compliance evaluator for Readily Take-Home assignment.
Replaces LLM evaluation for local testing.
"""

import random
from typing import List, Dict


def mock_evaluate_compliance(questions: List[str], policy_text: str) -> List[Dict]:
    """
    Generates mock evaluation results for each question.

    Parameters
    ----------
    questions : list[str]
        Extracted audit questions.
    policy_text : str
        Combined policy text from all policy PDFs.

    Returns
    -------
    list[dict]
        Each dict contains:
        {
            "question": str,
            "requirement_met": bool,
            "evidence": str
        }
    """

    # Split policy text into fake chunks for demonstration
    sample_evidence = policy_text[:1000]

    results = []
    for q in questions:
        met = random.choice([True, False])  # mock random result
        evidence = sample_evidence if met else "No relevant evidence found in provided policy text."
        results.append({
            "question": q,
            "requirement_met": met,
            "evidence": evidence
        })

    return results


if __name__ == "__main__":
    # Simple test
    questions = [
        "Does the P&P state that the MCP must respond to retrospective requests no longer than 14 calendar days from receipt?",
        "Is there a policy for member grievance tracking?"
    ]
    fake_policy_text = "This is a mock policy text that simulates extracted content from uploaded PDF documents."
    results = mock_evaluate_compliance(questions, fake_policy_text)

    for r in results:
        print(f"\nQuestion: {r['question']}")
        print(f"Requirement Met: {r['requirement_met']}")
        print(f"Evidence: {r['evidence'][:150]}...")
