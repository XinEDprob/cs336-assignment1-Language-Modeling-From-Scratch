# Modern Language Modeling from Scratch

A 16:9 LaTeX Beamer presentation based on the implementation in `cs336_basics/`.

- `modern_llm_from_scratch.pdf`: ready-to-present slides.
- `modern_llm_from_scratch.tex`: editable, standalone LaTeX source.
- `speaker_notes.md`: slide-by-slide explanations, code pointers, and a 30-minute schedule.

Present slides 1–22 in 27 minutes 30 seconds, then allow 2 minutes 30 seconds for discussion. Slides 23–28 are optional backup material and references. The assumed audience knows Python, matrix multiplication, and the idea of gradient descent, but need not know tokenizers or Transformers.

## Build

From the repository root:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=slides slides/modern_llm_from_scratch.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=slides slides/modern_llm_from_scratch.tex
```

Or upload the `.tex` file to Overleaf and select pdfLaTeX. No external image assets, bibliography processor, shell escape, or nonstandard fonts are required. The source uses Beamer, Latin Modern, AMS math, booktabs, and listings.

Equations and concise pseudocode explain the mechanisms. Implementation file references remain in the speaker notes. The slides omit repository names and code-source footers. External comparisons link to the original papers on the relevant slides and in the reference slides.

The deck describes the inspected code, including its implementation limitations. The 35,412,480-parameter backup example is a calculated configuration, not evidence of a completed training run. No model quality or benchmark results are claimed. The repository's Python files were not changed.
