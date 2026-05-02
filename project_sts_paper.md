---
name: STS Research Paper — Federated BP-Free Learning
description: Student's STS (Regeneron Science Talent Search) paper on convergence analysis of backpropagation-free federated learning with auxiliary networks. Co-advised by Dr. Chao Huang at Montclair State University.
type: project
originSessionId: 96d82932-cd1f-4429-a6ab-9fd3be8f2298
---
## Project Overview

**Student background:** Has studied split neural network learning for one year under Dr. Chao Huang (Montclair State University). Co-authored a paper on "Back Propagation-Free with Auxiliary Networks" — student did the convergence analysis (math), co-author did programming.

**STS project title:** *Convergence Analysis of Backpropagation-Free Federated Learning via Auxiliary Networks*

**Core contribution:** First convergence guarantees for BP-free auxiliary network training under federated, non-IID data conditions.

**Why:** Natural extension of student's existing BP-free convergence work into federated settings (privacy-preserving ML across hospitals/devices). Strong math story + real-world application.

---

## Main Theorem (Theorem 1)

Under 4 assumptions (L-smoothness, bounded aux error β, bounded gradients G, bounded heterogeneity σ_h), with η = c/(K√T):

> min_t ‖∇F(wₜ)‖² ≤ O(1/√T) + O(β²) + O(σ_h²)

- O(1/√T): standard convergence rate — vanishes with rounds
- O(β²): auxiliary network quality — controllable by design
- O(σ_h²): data heterogeneity — irreducible floor from non-IID data

Recovers classical FedAvg convergence when β → 0.

---

## Four Lemmas (Proof Structure)

- **Lemma 1** (One-Step Descent): F(w_{t+1}) ≤ F(w_t) − (η/4)‖∇F‖² + Cβ²  [condition: η ≤ 1/(4L)]
- **Lemma 2** (Client Drift): ‖w^(i)_{t,K} − w_t‖² ≤ K²η²(G+β)²
- **Lemma 3** (Aggregation Error): ‖Ĝ_t − ∇F(w_t)‖² ≤ L²K²η²(G+β)²/2 + 2β² + 2σ_h²
- **Lemma 4** (Telescoping): Sums descent over T rounds → Main Theorem

---

## Paper Sections — ALL DRAFTED

| Section | Status | Notes |
|---------|--------|-------|
| 1. Introduction | ✅ Done | Motivates BP-free federated learning, states 3 contributions |
| 2. Related Work | ✅ Done | BP-free learning, FedAvg convergence, split networks, inexact gradients |
| 3. Setup & Assumptions | ✅ Done | 4 formal assumptions, update rule, aggregation rule |
| 4. Lemmas 1–4 + Theorem 1 | ✅ Done | Full proofs with all steps |
| 5. Design Implications | ✅ Done | How to pick β, K, η for target ε; Corollaries 5.1 & 5.2 |
| 6. Numerical Experiments | ✅ Done | 4 experiments (3 MNIST + 1 real medical); all PyTorch code written |
| 7. Conclusion | ✅ Done | Two paragraphs; references medical validation and MIMIC-IV future work |

**Paper file:** `/Users/zqin/yan/sts_paper_draft.html` — open in browser, Cmd+P to save as PDF.

---

## Experiment Code (PyTorch) — Written, Not Yet Run

All three files use MNIST, N=10 clients, FedAvg aggregation, BP-free local training.

- **experiment1_convergence.py** — Validates O(1/√T) rate. Log-log slope should be ≈ −0.50.
- **experiment2_beta_floor.py** — Varies AUX_WIDTHS=[4,8,16,32,64,128]. Log-log slope of final_norm vs β² should be ≈ +1.00.
- **experiment3_heterogeneity.py** — Varies ALPHAS=[0.05,0.1,0.5,1.0,5.0,100.0] with weak (w=8) vs strong (w=128) aux nets. Gap between curves closes at low α (high σ_h).
- **experiment4_medical.py** — Real UCI Heart Disease dataset, 4 actual hospital sites (Cleveland USA, Hungarian, Long Beach VA USA, Zürich Switzerland). N=4 natural federated clients, 13 clinical features, binary classification. Compares: Centralized (upper bound), Local only (lower bound), FedAvg (β=0), BP-Free (ours). Measures σ_h from real inter-hospital population differences. Produces Figure 4 with 3 panels: AUC over rounds, gradient norm convergence vs theoretical floor, β̂ vs σ_h tracking.

**To run:**
```bash
pip install torch torchvision matplotlib numpy scikit-learn
python experiment1_convergence.py   # ~20 min CPU
python experiment2_beta_floor.py    # ~60 min CPU
python experiment3_heterogeneity.py # ~90 min CPU
python experiment4_medical.py       # ~5 min CPU (small dataset)
```

---

## Key Design Insights (Section 5)

- To achieve ε-accuracy: need β ≤ √(ε/2C₁) AND T ≥ O(1/ε²) rounds
- Optimal local steps: K* = (σ_h² + β²)^(1/2) / (Lη(G+β))
- When β² < σ_h²: improving aux net yields NO further benefit — heterogeneity is binding
- Privacy advantage: no gradients cross client-server boundary (reduces gradient inversion attack surface)

---

## Pending Tasks

1. **Run all 4 experiment scripts** and collect figures (Figures 1–4)
2. **Compile BibTeX references** (McMahan 2017, Li 2020, Huang 2025, Hinton 2022, Nøkland 2016, Karimireddy 2020, Zhao 2018, Vepakomma 2018, Belilovsky 2019, Nøkland & Eidnes 2019, Lillicrap 2016, Ajalloeian & Stich 2020, Detrano 1989)
3. **Write appendix** (optional — full proof details)
4. **STS application essays** — research summary (250 words, use abstract), significance essay (privacy in healthcare angle)
5. **Mentor letter** from Dr. Chao Huang

## STS Deadline
Mid-October (senior year). Advise student to confirm exact date at regeneron.org/sts.

---

## Why:** Extension of student's existing BP-free paper to federated setting. Novel math (first convergence guarantee for this setting). Real application (medical privacy). Strong mentor already in place. STS favors deep theory over broad survey.

## How to apply:** Resume this paper directly — all sections drafted. Next step is running experiments and assembling final PDF.
