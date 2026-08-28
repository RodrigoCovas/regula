# Mode is a per-run choice

Mode used to be boot-time configuration: `REGULA_MODE` pinned the process, and the canonical-scenario-id dispatch (ADR-0005) decided what Demo mode served. The frontend now offers a Demo/Live toggle, so mode is chosen per analysis request: every request carries `mode` explicitly, `REGULA_MODE` survives only as the server-side default when a request omits it (default Demo), and `live_eval` keeps pinning the mode per run regardless of boot configuration. ADR-0005 is unchanged — Demo mode still serves only the exact canonical Scenario, never derived ids.

## Consequences

- The backend boots without a Live-capable configuration: a missing provider key is a Readiness gap surfaced per request (and via the readiness endpoint), not a boot refusal in the default mode.
- API clients that omit `mode` get the server default, keeping curl examples stable.
