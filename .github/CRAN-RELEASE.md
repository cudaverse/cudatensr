# CRAN release checklist

This checklist separates a reproducible candidate from the external CRAN
submission and acceptance steps. A green ordinary package check is necessary,
but it is not by itself a release.

## Candidate

- [ ] Confirm that `DESCRIPTION`, `NEWS.md`, public documentation, and examples
      describe the candidate exactly.
- [ ] Confirm that the package name conflicts with neither current nor archived
      CRAN packages nor current Bioconductor packages.
- [ ] Run the manually dispatched `cran-readiness` workflow at the exact
      candidate commit.
- [ ] Require zero errors and warnings from the full R-devel
      `R CMD check --as-cran`, including reference-manual generation; review
      every note and allow only the expected `New submission` note.
- [ ] Review every URL and spelling result rather than suppressing unexpected
      findings.
- [ ] Review the uploaded source tarball and `00check.log`; submit that exact
      tarball without rebuilding it from a different commit.
- [ ] Check the same tarball with Win-builder R-devel before the first
      submission.

## Submission

- [ ] Read the current CRAN repository policy and submission checklist.
- [ ] Upload the exact verified source tarball through the CRAN submission
      form.
- [ ] Accept the confirmation email sent to the `DESCRIPTION` maintainer.
- [ ] Do not submit another build while this candidate is pending.

## Acceptance

- [ ] Verify the package and check-results pages on CRAN.
- [ ] Tag the accepted commit with the published version and create the matching
      GitHub release.
- [ ] Update the cudaverse compatibility table with the accepted version and
      commit.
- [ ] Only then prepare `cudasparsr` and `cudalearnr` without development
      `Remotes`; their strong `cudatensr` dependency must resolve from a
      standard repository.
