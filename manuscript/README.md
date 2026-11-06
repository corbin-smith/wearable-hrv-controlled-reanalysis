# Manuscript and supplement builders

These scripts build the manuscript and its Supplementary Material from the results in `../outputs`. Every number in the text is read from a results file rather than typed in.

| Folder | Contents |
|---|---|
| `main/` | `v12_template.md` (the manuscript text, with placeholders for every number), `v12_facts.py` (reads each number from `outputs/` and records its source), `v12_references.md`, `build_v12.py` and `docx_build.js` (assemble the Word file, in the draft or the Frontiers layout), `numcite.py` (numbered citations), `supplement_items.py` (the supplement's tables, figures and sections, numbered by first citation), `check_v12.py` (checks on the built file) |
| `supplement/` | `s6_template.md`, `s6_tables*.py` (the supplementary tables, built and in places computed from `outputs/`), `s6_facts.py`, `s6_common.py`, `build_s6.py`, `docx_s6.js`, `check_s6.py`, `inventory_s6.py` |

## Requirements

- Python with the packages in `../requirements.txt`.
- Node.js with the `docx` package (`npm install -g docx`; tested with docx 9.7.1).
- pandoc, which the checks use to read the built files back.

## Build

From the repository folder:

```
bash manuscript/build.sh
```

This writes the following files. Each build ends with its checks, which must report `0 check(s) failed`.

- `manuscript/main/build/manuscript_v12_draft.docx`
- `manuscript/main/build_frontiers/manuscript_v12_frontiers.docx`
- `manuscript/supplement/build/Supplementary_Material_v6.docx`

`main/facts/v12_facts_registry.csv` lists every number in the text with the file it came from.

Text marked `[[FLAG: ...]]` in the templates is shown highlighted in the built documents; it marks items still to be completed before submission.
