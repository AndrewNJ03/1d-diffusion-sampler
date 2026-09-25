# `latex_paper/`: ANS M&C 2027 paper

Source for the conference paper on gradient-enhanced GPR for POD-based ROMs,
written with the ANS topical-meeting class. The math in the paper is the math
documented in the code READMEs; this file maps each part of the paper to the
code and outputs behind it.

| File | Role |
|---|---|
| `main.tex` | Paper source |
| `main.bib` | Bibliography (cited entries only) |
| `ANStopical.cls`, `ANStopical.bst` | ANS class and BibTeX style |
| `figures/` | Figures, copied in from the study directories so the paper builds on its own |
| `main.pdf` and `main.{aux,bbl,blg,log,out}` | Last build and its intermediate files |

## Build

```bash
cd latex_paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The class needs several packages outside a minimal TeX install
(`titling`, `titlesec`, `enumitem`, `cleveref`, `authblk`, `lastpage`,
`newtxtext`/`newtxmath`, among others). On Debian or Ubuntu,
`texlive-latex-extra` and `texlive-fonts-extra` provide them. The page
count is on the last line of `main.log` (`Output written on main.pdf (N pages ...)`).

## Where each section comes from

| Paper section | Content | Code | Numbers / figures |
|---|---|---|---|
| §2 Problem setting, sandbox | FV diffusion model, LHS, QoI mask, POD | `common/` ([README](../common/README.md)) | |
| §3 Adjoint sensitivities | $A^\top\lambda_k = H^\top\psi_k$, per-parameter formulas, identity and FD checks | `common/adjoint_solver.py`, `sklearn_implementation/adjoint_stage.py`, `void_teflon_geometry/adjoint_geometry.py` | |
| §4 GE-GPR | Joint value/derivative covariance, augmented kernel, scaling | `sklearn_implementation/ge_gpr.py` ([README §4](../sklearn_implementation/README.md#4-gradient-enhanced-gpr-ge_gprpy)) | |
| §5.1 Material-layer case | $D$, $\Sigma_a$, $q$ one at a time | `gpr_vs_gegpr_comparison/run_comparison.py` | `gpr_vs_gegpr_comparison/summary_table.csv` |
| §5.2 Geometry case | $t_v$, $a$ one at a time and both together | `gpr_vs_gegpr_comparison/`, `void_teflon_geometry/run_2param_geometry.py` | `summary_table.csv`, `void_teflon_geometry/output_graphs_geometry_2param/` |
| §5.3 Combined case | all 5 active | `combined_study/run_combined.py` | `combined_vs_one_at_a_time.csv`; `figures/combined_learning_curve.png` ← `combined_study/output_graphs_combined/gegpr_learning_curve.png` |
| §5.4 Repeatability | 5 LHS designs | `combined_study/run_combined_seeds.py` | `combined_seed_repeats.csv`; `figures/combined_seed_repeats.png` ← `combined_study/combined_seed_repeats.png` |
| §5.5 SCEPTRE | Standard-GPR learning curves on SGEMP $q_\text{net}$ | `examples/sceptre_sgemp_charge/`, `combined_study/cep_sgemp_analysis/` | `output_*/learning_curve.csv` |

The figures are copies, not links, so after rerunning a study copy the new
PNG over the one in `figures/`.
