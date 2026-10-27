# Security

Auradream is a single-user application meant to run on your own
machine. The server binds to `127.0.0.1` by default, checks every request's
`Host` header against `AURADREAM_ALLOWED_HOSTS`, refuses a
state-changing request whose `Origin` names another site (requests without an
`Origin`, such as the CLI or curl, are allowed), and has **no login**. Exposing it on a network gives everyone who can reach it full
control of the gallery and of the model downloads; see the warning under
[Settings](README.md#settings).

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/matt439/Auradream/security/advisories/new)
rather than in a public issue. Include the version (commit), the steps to
reproduce and what an attacker gains.

Useful reports include a way for a web page, a crafted PNG, gallery bundle,
LoRA name or mask to read or write files outside the app's folders, run code,
or act on the server from another origin while it is bound to loopback.
Issues that need the server deliberately exposed without a proxy, or a
malicious checkpoint you chose to download, are expected behaviour.

This is a personal project maintained on a best-effort basis; there is no
fixed response time.
