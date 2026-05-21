# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email your findings to the maintainer via [GitHub Security Advisories](https://github.com/xuwenyao/agentic-os-core/security/advisories/new).
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Affected versions
   - Possible mitigation (if any)

## Response Timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 7 days
- **Fix / disclosure**: depends on severity, typically within 30 days

## Scope

This project is a **pure Python library with zero external dependencies**. The attack surface is minimal:

- No network I/O in core code
- No file system writes (user controls paths)
- No code execution or eval
- Plugin interfaces are abstract — security of plugin implementations is the user's responsibility
