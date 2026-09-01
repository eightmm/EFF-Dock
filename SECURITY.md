# Security Policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
private vulnerability reporting for this repository when it is available. If
that channel is unavailable, open a minimal issue asking the maintainers for a
private contact channel without including exploit details, credentials, or
sensitive data.

Include the affected revision, environment, reproduction conditions, impact,
and any proposed mitigation. Reports concerning unsafe model deserialization,
untrusted structure parsing, command execution, path traversal, or accidental
credential/data exposure are especially useful.

## Scope

Only the current `main` branch is maintained. Model predictions are research
outputs, not medical advice or a security boundary. Third-party datasets,
checkpoints, CUDA libraries, and cluster infrastructure retain their own
security and licensing policies.
