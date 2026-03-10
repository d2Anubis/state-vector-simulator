# Feature Proposal: Strict Circuit Schema Validator + Resource Guardrails

## Feature 1: Strict Circuit Schema Validator
- Validate JSON circuit payloads before execution.
- Fail early with clear ValueError messages for invalid gates, qubit/clbit ranges, and measurement mappings.

## Feature 2: Resource Guardrails
- Add configurable limits (e.g., max_qubits, max_state_bytes).
- Reject oversized circuits before allocation to prevent memory exhaustion.

## Why This Fits
- Improves reliability and safety for CLI + library usage.
- Reduces runtime crashes from malformed inputs or oversized requests.

## Feedback Requested
1. Should strict validation be opt-in first (`strict=True`) or default-on?
2. What default qubit/memory limits should be used?
