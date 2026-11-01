# Cosine and Tversky simulation reproducibility archive preparation

This folder is prepared for author review and a subsequent repository deposit. It has not been deposited or assigned a DOI. Yili Lin has authorized MIT for the independently owned code and CC BY 4.0 for releasable evaluation data. Public data distribution remains subject to confirmation of source rights. Do not apply a blanket licence to third-party SNOMED content.

## Reproduce the reported statistical results
Run `python reproduce_from_counts.py` with Python 3.6+, pandas and numpy. It writes reproduced/ and checks all summaries and confidence intervals against expected/. No SNOMED release file, account or terminology service is required for this counts-based calculation.

The full ontology pipeline is in full_ontology_analysis.ipynb. Its source inputs, source hashes, omission parameters and software versions are recorded in run_metadata.json. Obtain SNOMED CT US Edition RF2 Relationship Full 20260301 through lawful access, and obtain the reviewed seed-concept workbook from the authors under the applicable access conditions. The licensed feature sets, RF2 file and seed workbook are not included in this unrestricted archive candidate.

Read DATA_DICTIONARY.md for columns and limitations. All results are based on repeated local-neighborhood simulations, not real patient expression annotations. Discuss the licensed-input exception with the journal before finalizing the deposit and data availability statement.

## Scope and licences
This standalone study does not use or depend on DEMI. Software files (*.py and *.ipynb) are MIT licensed; author-generated data and results are covered by DATA_LICENSE.md. Third-party inputs retain their own terms.

## Publication status
Version 1.0.0 is prepared for deposit; the associated manuscript has not been submitted or accepted. A software/data DOI does not imply journal publication.
