# Security

AuraDream is a single-user application meant to run on your own
machine. The server binds to `127.0.0.1` by default, checks every request's
`Host` header against `AURADREAM_ALLOWED_HOSTS`, refuses a
state-changing request whose `Origin` names another site (requests without an
`Origin`, such as the CLI or curl, are allowed). The optional
`AURADREAM_PASSCODE` asks for a shared passcode when the bind address or an
allowed host reaches beyond loopback; the local browser and a proxy's
loopback requests then sign in too. Without it, everyone who can reach the
server has full control of the gallery and model downloads. There are no
separate accounts or permissions.

On plain http, the passcode, its session cookie, prompts and pictures travel
unencrypted; someone who captures network traffic can read them and reuse
the sign-in. Use a trusted network and HTTPS or a VPN for transport privacy,
and never forward the port to the internet. Do not reuse another password.
See [Settings](README.md#settings) and the
[Windows phone recipe](docs/setup.md#from-a-phone-windows).

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/matt439/AuraDream/security/advisories/new)
rather than in a public issue. Include the version (commit), the steps to
reproduce and what an attacker gains.

Useful reports include a way for a web page, a crafted PNG, gallery bundle,
LoRA name or mask to read or write files outside the app's folders, run code,
or act on the server from another origin while it is bound to loopback.
A bypass of the enabled passcode gate, its session verification or its
wrong-passcode limiter is also in scope. Access to a deliberately exposed
server with no passcode, capture of plain-http traffic, and a malicious
checkpoint you chose to download are expected consequences of those choices.

This is a personal project maintained on a best-effort basis; there is no
fixed response time.
