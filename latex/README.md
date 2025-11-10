# BrainTumNetV2 Paper - LaTeX Source

This directory contains the LaTeX source for the BrainTumNetV2 research paper prepared for Springer LNCS/LNAI conference submission.

## Paper Structure

- **main.tex**: Main document with document class, packages, and structure
- **sections/introduction.tex**: Introduction section
- **sections/related_work.tex**: Related work and literature review
- **sections/methodology.tex**: Detailed methodology (architecture, loss functions, training)
- **sections/experiments.tex**: Experiments, results, and analysis
- **sections/conclusion.tex**: Conclusion and future work
- **references.bib**: BibTeX bibliography file

## Requirements

### LaTeX Distribution

You need a LaTeX distribution installed:
- **Windows**: MiKTeX or TeX Live
- **macOS**: MacTeX
- **Linux**: TeX Live

### Required Packages

The following LaTeX packages are required (usually included in full distributions):
- `llncs` (Springer LNCS document class)
- `graphicx` (for figures)
- `amsmath`, `amssymb` (mathematical notation)
- `algorithm`, `algorithmic` (algorithms)
- `booktabs` (professional tables)
- `multirow` (table formatting)
- `xcolor` (colors)
- `hyperref` (hyperlinks)

### Getting LNCS Class

Download the Springer LNCS package from:
https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines

Extract and place `llncs.cls` in the same directory as `main.tex`.

## Compilation

### Using pdfLaTeX (Recommended)

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The double run after BibTeX ensures all references are correctly resolved.

### Using latexmk (Automated)

```bash
cd latex
latexmk -pdf main.tex
```

This automatically runs all necessary compilations.

### Using Overleaf

1. Create a new project on Overleaf
2. Upload all `.tex` files, `.bib` file, and `llncs.cls`
3. Set compiler to `pdfLaTeX`
4. Compile

## TODO: Complete After Training

### Results to Fill In

The paper currently contains placeholder `[TODO]` markers for results that need to be filled in after training completes:

1. **Table 1 (Comparison with SOTA)**: Fill in actual Dice, IoU scores for WT, TC, ED
2. **Table 2 (Ablation Study)**: Fill in ablation results showing contribution of each component
3. **Table 3 (Classification Performance)**: Fill in accuracy, precision, recall, F1 for tumor grading
4. **Table 4 (Computational Efficiency)**: Measure actual FLOPs, training time, inference time

### Figures to Create

Create the following figures and save as PDF or PNG:

1. **Figure 1: Architecture Diagram** (`fig:architecture`)
   - Overall BrainTumNetV2 architecture
   - Show encoder, transformer bottleneck, decoder, multi-task heads
   - Use diagrams/architecture_diagram.pdf or similar tool (draw.io, TikZ, etc.)

2. **Figure 2: Qualitative Results** (`fig:qualitative`)
   - 3-4 example segmentations (HGG and LGG cases)
   - Layout: Input (T1CE) | Ground Truth | Prediction | Error Map
   - Color coding: Red=TC, Green=ED, Blue=Background
   - Save as figures/qualitative_results.pdf

3. **Figure 3: Attention Visualization** (`fig:attention`)
   - CBAM attention maps at different decoder levels
   - Soft masks from adaptive masked transformer
   - ROI masks for classification
   - Save as figures/attention_maps.pdf

4. **Figure 4: Feature Visualization** (`fig:features`)
   - t-SNE/UMAP of bottleneck features
   - Color by tumor grade, size, region
   - Save as figures/feature_tsne.pdf

5. **Figure 5: Training Curves** (`fig:loss_curves`)
   - Total loss, component losses over epochs
   - Validation Dice scores (WT, TC, ED) over epochs
   - Learning rate schedule
   - Save as figures/training_curves.pdf

6. **Figure 6: Per-Region Analysis** (`fig:per_region`)
   - Box plots of Dice scores for WT, TC, ED
   - Stratified by tumor size and grade
   - Save as figures/per_region_analysis.pdf

### How to Add Figures

1. Create figures using Python/matplotlib or other tools
2. Save as PDF (vector) or high-res PNG (300+ DPI)
3. Place in `latex/figures/` directory
4. Uncomment figure code in sections/experiments.tex
5. Update figure filenames to match your saved files

Example:
```latex
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/architecture_diagram.pdf}
\caption{Overall architecture of BrainTumNetV2...}
\label{fig:architecture}
\end{figure}
```

## Paper Specifications

- **Format**: Springer LNCS/LNAI
- **Target Length**: 12-15 pages
- **Language**: English
- **Submission**: Camera-ready PDF

Current draft is structured to fit within 12-15 pages when figures are added.

## Checklist Before Submission

- [ ] Fill in all `[TODO]` placeholders with actual results
- [ ] Create and add all required figures
- [ ] Update author names and affiliations in main.tex
- [ ] Add acknowledgments section with funding, resources, dataset credits
- [ ] Verify all citations are correct and complete
- [ ] Check mathematical notation consistency
- [ ] Proofread for grammar and spelling
- [ ] Ensure figures are high quality (vector or 300+ DPI)
- [ ] Verify table formatting follows LNCS style
- [ ] Check that references follow LNCS bibliography style
- [ ] Generate final PDF and verify it compiles without errors
- [ ] Check page count (should be 12-15 pages)
- [ ] Verify all cross-references work (sections, figures, tables, equations)

## Notes

### Methodology Section

The methodology section (Section 3) contains very detailed mathematical descriptions based on the actual code implementation:

- **Architecture**: Exact layer specifications, channel dimensions, activation functions
- **Loss Functions**: Complete mathematical formulations with all hyperparameters
- **Training**: Precise optimizer settings, learning rate schedule, data augmentation

This level of detail ensures full reproducibility.

### Code-Based Accuracy

All technical details are extracted directly from the code:
- `braintumnet_v2.py`: Main architecture
- `seg_unet_v2.py`: Segmentation network (478 lines analyzed)
- `masked_transformer.py`: Transformer bottleneck
- `cbam.py`: Attention module
- `t_inception.py`: Classification network
- `losses/combined.py`: Ultimate loss function (425 lines)
- `losses/multiclass.py`, `losses/iou.py`, `losses/boundary.py`: Individual loss components
- `metrics/multiclass.py`: Evaluation metrics
- `phase2_a100.yaml`: Training configuration

This ensures all formulas match the actual implementation.

## Contact

For questions about the paper or code implementation:
- Email: [Your email]
- GitHub: [Your repo URL when public]

## License

[TODO: Add license information]
