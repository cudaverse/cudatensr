## New submission

This is the first CRAN submission of `cudatensr`. The package has no strong
dependencies outside base R. Its optional `torch` backend is listed in
`Suggests`; all examples and tests that require it are conditional.

An explicit `device = "cuda"` request fails with an actionable error when a
CUDA-enabled R `torch` installation is unavailable. The default
`device = "auto"` path uses the portable base R backend and records that
selection in the returned provenance.

## Test environments

- Windows 11, R 4.6.0 (local)
- Windows Server 2022, R-devel r90304 UCRT (Win-builder)
- Windows, current R release (GitHub Actions)
- macOS, current R release (GitHub Actions)
- Ubuntu, current R release (GitHub Actions)
- Ubuntu, R-devel (GitHub Actions)

## R CMD check results

0 errors | 0 warnings | 1 note

The incoming-feasibility note contains:

- This is a new submission.
- `CUDA` is listed as a possibly misspelled word in `DESCRIPTION`. CUDA is the
  proper name of NVIDIA's parallel-computing platform and programming model.

The local Windows check used the source tarball produced by `R CMD build` and
`R CMD check --as-cran`, with all suggested packages installed, including
`torch` 0.17.0. The machine has no usable CUDA device, so the one explicitly
hardware-only transfer test was skipped while the CPU backend and unavailable
CUDA diagnostics were exercised. PDF and HTML reference-manual checks passed.

The manually dispatched `cran-readiness` workflow checks spelling and URLs,
builds one source tarball with the current R release, records its SHA-256, and
hands that exact tarball to a separate R-devel job. The R-devel job runs a full
CRAN-style check including the reference manual and retains the candidate,
source provenance, manual, and check logs together.

The exact 0.2.0 candidate from commit
`1b7b412707224765da5cdc679cc66cf2ef4cc56f` was built with R 4.6.1 and has
SHA-256
`7a1c8c0bbe6f0d404f986770f860ef1408e313a3c304d036896f10cf5623002a`.
Win-builder installed, built, and checked that tarball under Windows R-devel.
All installation, code, documentation, example, test, vignette, and manual
checks passed; its status was the single incoming-feasibility note explained
above.

## Downstream dependencies

There are currently no CRAN reverse dependencies because this is a new
submission. Other cudaverse packages depend on `cudatensr` in their development
sources, but they will not be submitted until this foundation package is
available from CRAN.
