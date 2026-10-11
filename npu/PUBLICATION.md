# Publication provenance

This checkout contains the OpenPI NPU adaptation and the 2026-10-11 four-card
measurements. The original workspace and raw evidence remain unchanged locally.
Published evidence redacts the worker hostname, private OBS prefix, local workspace
and SSH connection identifiers. Measurements, tensor statistics, failures and exit
codes are preserved. Server cache paths are historical artifact locators, not
credentials or portable defaults. Weights, datasets, environments, remote access
helpers, upload logs and profiler binaries are excluded.

`SOURCE_MANIFEST.json` describes historical `source008.tar.gz`, not this release.
Hashes recorded inside execution receipts identify the original executed files.
`publication_manifest.json` records the original and released file hashes separately.
The release adds `--weights`/`--data` to the profiling and optimization scripts,
removes the fixed venv from `environment.sh`, and makes the dataset source label
configurable. These packaging changes have local syntax/structural checks; the
published revision has not been rerun on NPU. The model/optimizer math is unchanged.

The upstream README (with trailing whitespace removed) is in `README_UPSTREAM.md`. Upstream Git history,
LICENSE and submodule references are retained. Existing upstream GitHub workflows
are manual-only because their CI environment is not the measured NPU environment.
