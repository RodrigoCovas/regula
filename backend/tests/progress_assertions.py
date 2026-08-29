"""Shared progress assertions for the registry and endpoint test files.

Both the pure registry (test_progress.py) and the HTTP seam
(test_progress_endpoint.py) assert the same two contracts — the five
workflow phases in agent order with informative messages, and the
Not-available reply for an unknown request id — so the assertions live
here once instead of twice.
"""

from test_analyze import assert_not_available_shape

PHASE_ORDER = ["planner", "researcher", "verifier", "proposer", "summarizer"]

_MESSAGE_CONTENT_CHECKS = [
    ("target", "the Planner's message names research targets"),
    ("evidence", "the Researcher's message names Evidence"),
    ("claim", "the Verifier's message names Claims"),
    ("action", "the Proposer's message names Actions"),
    ("relevance", "the Summarizer's message names Provision relevance"),
]


def assert_phases_in_agent_order(reported):
    """The five workflow agents report in order: phases are exactly
    Planner → Researcher → Verifier → Proposer → Summarizer, every message is
    non-empty, and each phase's message names what the agent does.

    ``reported`` is a list of (phase, message) pairs in report order.
    """
    assert [phase for phase, _ in reported] == PHASE_ORDER
    for _, message in reported:
        assert message, "every phase carries an informative message"
    messages = " ".join(message for _, message in reported).lower()
    for needle, reason in _MESSAGE_CONTENT_CHECKS:
        assert needle in messages, reason


def assert_unknown_request_reply(data, request_id):
    """The unknown-id reply: the Not-available sibling shape, naming the
    id, the TTL expiry as a cause, and how to proceed."""
    assert_not_available_shape(data)
    actions = " ".join(data["answer"]["actions"]).lower()
    assert request_id.lower() in actions, "the reply names the unknown id"
    assert "expired" in actions, "the reply names the TTL expiry as a cause"
    assert "re-submit" in actions, "the reply says how to proceed"
    assert any("english-only" in line.lower() for line in data["known_limitations"])
