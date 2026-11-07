# Security policy

## Supported versions

I fix security problems on `main` and ship them in the next release.

| Version | Supported |
|---|---|
| Latest release | Yes |
| `main` | Yes |
| Older releases | No |

## Reporting a vulnerability

Please report security problems privately through GitHub's private vulnerability reporting. On the repository page, open the **Security and quality** tab and click **Report a vulnerability**, or go straight to the [advisory form](https://github.com/mertkayacs/tholos/security/advisories/new). Keep the problem out of public issues until a fix is released.

Please include:

- the Tholos version or commit you run,
- how you run it: uv or Docker, the operating system and the model server,
- the steps to reproduce it and what an attacker gains,
- a proof of concept, logs or screenshots, with API keys, tokens and private data removed.

I reply in the advisory, work on the fix there with you, and credit you in the published advisory if you would like to be named.

## What counts as a security problem

Tholos runs on your own machine and relies on these protections:

- it listens on loopback by default and asks for an access token on any other address,
- on loopback it accepts only local Host headers, and every change needs a same-origin request with a CSRF token,
- web fetches go only to public addresses over HTTP or HTTPS on ports 80 and 443, with limits on redirects, time and size,
- an approval applies to the exact tool and arguments shown, and only the owner can change rules,
- the database, which holds API keys and the Telegram token, is readable only by your user.

A way around any of these is a security problem. A model that makes a wrong change it was allowed to make is a bug, so please open a normal issue for that.
