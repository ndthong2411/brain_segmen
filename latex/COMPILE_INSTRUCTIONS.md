# Compilation Instructions - LNCS Format

## Prerequisites

The paper follows **Springer LNCS/LNAI** (Lecture Notes in Computer Science) format.

Required files (already in `template/` folder):
- `llncs.cls` - LNCS document class (version 2.24)
- `splncs04.bst` - LNCS bibliography style

## Compilation Methods

### Method 1: Overleaf (Recommended - Easiest)

1. **Upload all files** to Overleaf:
   - `main.tex`
   - All `sections/*.tex` files
   - `references.bib`
   - **IMPORTANT**: `template/llncs.cls` and `template/splncs04.bst`

2. **Set compiler**: pdfLaTeX (default)

3. **Project structure** in Overleaf:
   ```
   main.tex
   references.bib
   sections/
     introduction.tex
     related_work.tex
     methodology.tex
     experiments.tex
     conclusion.tex
   template/
     llncs.cls        ← MUST have this
     splncs04.bst     ← MUST have this
   figures/           ← Add your figures here (when ready)
   ```

4. **Main document**: Set `main.tex` as main document

5. Click **Recompile**

### Method 2: Local LaTeX (Command Line)

#### Step 1: Copy class files to main latex folder

```bash
# On Windows (cmd)
copy template\llncs.cls .
copy template\splncs04.bst .

# On Windows (PowerShell)
Copy-Item template\llncs.cls .
Copy-Item template\splncs04.bst .

# On Linux/Mac
cp template/llncs.cls .
cp template/splncs04.bst .
```

#### Step 2: Compile with pdfLaTeX

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The double `pdflatex` run after `bibtex` is necessary to resolve all references.

### Method 3: LaTeXmk (Automated)

```bash
cd latex
latexmk -pdf main.tex
```

This automatically runs pdfLaTeX and BibTeX the correct number of times.

## Current Status

✅ **Structure**: Complete and follows LNCS format
✅ **Content**: All sections written (Introduction, Related Work, Methodology, Experiments, Conclusion)
✅ **Bibliography**: 25+ references in `references.bib` with `splncs04` style
✅ **Format**: Proper LNCS formatting with `\usepackage[T1]{fontenc}`, running heads, etc.

⚠️ **TODO**:
- Fill in actual experimental results (marked with `[TODO]` in experiments.tex)
- Create 6 figures (specifications provided in experiments.tex)
- Update author information in main.tex
- Add acknowledgments in conclusion.tex

## LNCS Format Compliance Checklist

✅ Document class: `\documentclass[runningheads]{llncs}`
✅ T1 font encoding: `\usepackage[T1]{fontenc}`
✅ Title with running head: `\titlerunning{...}`
✅ Author with ORCID: `\orcidID{...}`
✅ Author running head: `\authorrunning{...}`
✅ Abstract with keywords: `\keywords{... \and ... \and ...}`
✅ Bibliography style: `\bibliographystyle{splncs04}`
✅ Acknowledgments: `\begin{credits}\subsubsection{\ackname}...\end{credits}`
✅ Competing interests: `\subsubsection{\discintname}...`

## Expected Output

- **Format**: PDF following Springer LNCS style
- **Page count**: 12-15 pages (target range for LNCS conferences)
- **Font**: Computer Modern (LaTeX default), 10pt body text
- **Margins**: LNCS standard margins
- **Running heads**: Short title and author names on each page

## Troubleshooting

**Error: "llncs.cls not found"**
- Solution: Copy `template/llncs.cls` to the same directory as `main.tex`
- Or: Upload `template/llncs.cls` to Overleaf root

**Error: "splncs04.bst not found"**
- Solution: Copy `template/splncs04.bst` to the same directory as `main.tex`
- Or: Upload `template/splncs04.bst` to Overleaf root

**Bibliography not appearing**
- Make sure you run: pdflatex → bibtex → pdflatex → pdflatex
- Check that `references.bib` file exists and has no syntax errors

**Figures not appearing**
- Create `figures/` folder
- Add figure files (preferably PDF/EPS format for vector graphics, or high-res PNG)
- Update figure references in experiments.tex to remove `[TODO]` markers

## File Structure

```
latex/
├── main.tex                 ← Main document (compile this)
├── references.bib           ← Bibliography database
├── sections/
│   ├── introduction.tex     ← Section 1
│   ├── related_work.tex     ← Section 2
│   ├── methodology.tex      ← Section 3 (most detailed)
│   ├── experiments.tex      ← Section 4 (has TODO placeholders)
│   └── conclusion.tex       ← Section 5
├── template/
│   ├── llncs.cls           ← LNCS class file (REQUIRED)
│   ├── splncs04.bst        ← Bibliography style (REQUIRED)
│   ├── llncsdoc.pdf        ← LNCS documentation
│   ├── samplepaper.tex     ← LNCS sample/reference
│   └── readme.txt          ← LNCS package info
├── figures/                ← Create this, add your figures
├── COMPILE_INSTRUCTIONS.md ← This file
├── README.md               ← General paper information
├── TECHNICAL_SUMMARY.md    ← Quick reference for formulas
└── QUICK_START.md          ← Quick start guide
```

## Paper Sections Overview

1. **Introduction** (~1.5 pages): Problem, challenges, contributions
2. **Related Work** (~1 page): CNN segmentation, transformers, multi-task learning, loss functions
3. **Methodology** (~4 pages): Architecture details, loss functions, training
4. **Experiments** (~3-4 pages): Dataset, results tables (needs results), qualitative analysis (needs figures)
5. **Conclusion** (~1 page): Summary, limitations, future work

**Total**: ~12-15 pages when complete with figures and results

## Next Steps

1. **Complete training** to get actual performance numbers
2. **Fill TODO placeholders** in experiments.tex with real results
3. **Create figures** following specifications in experiments.tex
4. **Update author info** (name, institution, ORCID) in main.tex
5. **Add acknowledgments** in conclusion.tex
6. **Test compile** to ensure no errors
7. **Submit** to conference/journal
