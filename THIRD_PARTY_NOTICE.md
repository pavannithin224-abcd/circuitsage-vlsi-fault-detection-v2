# Third-party notice

This repository **redistributes** pinned snapshots of third-party RTL under `corpus/`.
This differs from the CircuitSage V1 release, which referenced upstream sources without
copying them. Each archive retains its **original upstream license**, and nothing in this
project's Apache-2.0 license alters, replaces, or relicenses upstream terms.

Apache-2.0 as applied here covers only contributions authored by Pavan Nithin: the stage
scripts, contracts, testbench wrappers, measurement code, and documentation.

## Redistributed designs

| design | upstream project | pinned revision | license |
|---|---|---|---|
| `opentitan_hmac_sha256` | [lowRISC/opentitan](https://github.com/lowRISC/opentitan) | `83fc48ed3a727399056772d12be8c7d4a8a276f0` | Apache-2.0 |
| `ibex_cpu` | [lowRISC/ibex](https://github.com/lowRISC/ibex) | `e9f55342edbd27e9e17a0e41b1c95a81abb5eac8` | Apache-2.0 |
| `picorv32_cpu` | [YosysHQ/picorv32](https://github.com/YosysHQ/picorv32) | `ef203c2b0a3fb793280f5114941416c425c5b461` | ISC |
| `secworks_aes` | [secworks/aes](https://github.com/secworks/aes) | `80dc4718e1dcbbdb4b0dd1bdb393d8f7b98981dc` | BSD-2-Clause |
| `secworks_chacha` | [secworks/chacha](https://github.com/secworks/chacha) | `7eaba360df9fed9fc2db98d5f3df81cf01e5b604` | BSD-2-Clause |
| `secworks_sha256` | [secworks/sha256](https://github.com/secworks/sha256) | `837c5cc396f001d18f2c765721c585716eb439ae` | BSD-2-Clause |

Archive integrity hashes are recorded in `corpus/acquired_family_registry_12c1c.csv`
(`tree_sha256`, `archive_sha256`). Verify before use.

All six licenses are permissive and mutually compatible; none is copyleft.

## Design NOT redistributed

| design | upstream project | license | reason |
|---|---|---|---|
| `serv_cpu` | [olofk/serv](https://github.com/olofk/serv) | ISC | **sealed evaluation holdout** |

`serv_cpu` was acquired as a sealed holdout and **deliberately never examined**. It is
excluded from this repository so the seal remains verifiable: a reader can confirm the
source was not available to the study. The exclusion is enforced in the packaging script
and verified three ways (filename scan, path scan, binary-payload scan).

Text references to `serv_cpu` appear throughout the audit chain. These are **records that it
stayed sealed**, not the design itself.

## Attribution requirements

Redistributing `corpus/` contents, in whole or part, requires honouring upstream terms:

- **Apache-2.0** (OpenTitan, Ibex) — preserve `LICENSE` and `NOTICE`, state changes made.
  The archives here are unmodified snapshots; all instrumentation lives in this project's
  own `rtl/` and `tb/`.
- **ISC** (PicoRV32) — preserve the copyright and permission notice.
- **BSD-2-Clause** (secworks AES, ChaCha, SHA-256) — preserve the copyright notice and the
  two-clause condition list.

The authoritative license text for each design is inside its archive. Where this summary and
an upstream license disagree, **the upstream license governs**.

## Tooling

Synthesis and simulation used the OSS CAD Suite, pinned to:

- Verilator `5.051 devel rev v5.050-222-gf6f6f8404 (mod)`
- Yosys `0.68+106 (git sha1 c92678eb2-dirty)`

These tools are **not redistributed here** and retain their own licenses (Verilator:
LGPL-3.0/Artistic-2.0; Yosys: ISC). Python dependencies likewise retain their respective
licenses.

## Trademarks

OpenTitan and lowRISC are marks of their respective owners. Use here is nominative — to
identify the designs studied — and implies no endorsement, affiliation, or sponsorship.

## Corrections

If you are an upstream maintainer and find an attribution error or a licensing concern,
please open an issue. Attribution problems will be treated as defects and corrected
promptly.
