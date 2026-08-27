# Derived scenario ids never trigger Demo mode; the canonical demo is offered explicitly

The frontend derives a Scenario's id from its description (content-word slug plus a hash suffix) because users never set ids by hand, and during grilling we briefly considered letting free text that resembles the canonical demo Scenario fall through to Demo mode — whether by exact-ish slug coincidence or by an embeddings-based semantic match. We decided Demo-mode dispatch stays an exact match on the canonical id, and the UI offers the canonical demo through an explicit "Try the demo scenario" button that sends the exact canonical inputs. Any free-text Scenario gets a derived id, which by construction never equals the canonical one, so in Demo mode it receives the honest Not-available response instead of a canned answer.

## Considered and rejected

- Semantic similarity trigger (embed the description, compare against the canonical Scenario): an embeddings-based trigger would make the keyless, LLM-free demo path depend on Ollama being up, and it re-introduces the silent-degradation the exact-match routing was chosen to forbid (main.py:495-497) — a near-demo description would silently get a canned demo answer.
- Derived-id whitelisting (special-case slugs that resemble "spanish-fintech-startup-uses-9e165169"): heuristic routing by another name; the explicit button makes the same convenience available without ambiguity about which path answered.

## Consequences

- Typing the canonical demo description by hand in Demo mode yields a Not-available response; the canonical demo is one button press away instead.
- The exact-match, non-heuristic dispatch in the backend stays untouched; the frontend owns the derivation and the explicit demo affordance.
