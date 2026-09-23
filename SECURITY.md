# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | yes       |
| < 2.0   | no        |

## Reporting a Vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report via:

- GitHub Security Advisories: https://github.com/MrModelOS/Glyph/security/advisories/new
- Or email the maintainers (see GitHub profile) with `[SECURITY]` in the subject.

Include:

- Description of the vulnerability
- Steps to reproduce
- Affected version / commit
- Any suggested fix

We aim to acknowledge reports within 3 business days and to provide a fix or mitigation timeline within 14 days. We will coordinate disclosure once a fix is available.

## Scope

This policy covers the `glyphc` compiler and runtime. Build-tool dependencies (gcc, Rust toolchain) are out of scope but please still report if Glyph's usage of them is insecure.
