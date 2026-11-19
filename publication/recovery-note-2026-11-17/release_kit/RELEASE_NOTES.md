# Local resource recovery note - {{VERSION}}

Companion assembled {{BUILD_DATE}} for note version 1.0.0, prepared 2026-11-17. No deposit on GitHub or Zenodo was performed by this assembly.

## Packaging correction v1.0.1

The earlier value 280 counted only original archived files from the three studies. It was not an outdated total for the whole package. VALIDATION.json now explicitly distinguishes 280 original archive files, 326 files listed in the package manifest, and 327 ZIP files including the manifest. The verifier checks these counts against both manifests automatically. PDF, scientific code and research data are unchanged; checksums and package verification reports have been regenerated.

## Included

- Russian research note with a separate title page, the full requested human / AI author line and a detailed contribution statement.
- Three complete archived studies from 2026-10-08: minimal observer, help timing, repeated help.
- Required working code, protocols, reporting and plotting scripts, selected tests and one short fixed-seed example.
- Source provenance, per-file SHA-256 checksums, numeric audit and recorded package validation.
- MIT for code; CC BY 4.0 for text and data, as chosen by Katia.
- Compact equations for resource flow and peer selection, a parameter table, clarified timing comparisons and post-hoc counterexample status, following the ChatGPT (Lyra) review. See REVIEW_RESPONSE.md.

The original scientific revision is `{{BASE}}`. The new note, attribution, assembly tools and package documentation are later additions. A public release should tag the commit that actually contains the new note and release assets, rather than attaching their provenance to an old commit that lacks them.

The package does not claim a new general architecture, a completed metastable cycle, or a statistically established advantage of periodic repeated help. It preserves the model limitations, exploratory comparisons and post-hoc selection of the local counterexample.

Exact checks executed for this package are in `VALIDATION.json`. Full experimental sweeps and all bootstrap intervals were not rerun during assembly.
