# STS High School Project — Decoupled Split Learning on Medical Imaging

Working plan for an STS (Science Talent Search) submission that applies
the **Decoupled Split Learning** algorithm (Zihad, Owino, Tang, Huang —
"Decoupled Split Learning via Auxiliary Loss") to multi-label medical
imaging data. Status: planning only. No experiments started yet.

Started: 2026-06-10.

---

## 1. Why this works as an STS project

The original DSL paper only tested CIFAR-10/100. Three reasons applying
DSL to medical data is a credible STS submission rather than a "swap the
dataset" exercise:

1. **Real gap.** No published evaluation of DSL on medical imaging.
2. **Coherent motivation.** Split learning was created for healthcare —
   hospitals can't share raw data, edge devices are bandwidth-limited.
   The 50% comm reduction maps directly to a clinical pain point.
3. **Real research questions emerge** when DSL meets medical data:
   multi-label heads, severe class imbalance, non-IID cross-hospital
   distribution. Each is a place the student's contribution can live.

What would make it competitive (vs. a science-fair poster):

- A real dataset, not MedMNIST.
- Non-IID / cross-institution simulation.
- Medical-specific failure analysis (rare-disease gap).
- At least one proposed modification with ablation.
- Clinical-deployment framing in the discussion.

---

## 2. Datasets

| Dataset | Size | Task | Why |
|---|---|---|---|
| **CheXpert** (primary) | 191K frontal X-rays | 14-class multi-label | Recognized benchmark, DUA-only access, multi-label exposes the aux-head design question |
| **ISIC 2019** (secondary) | ~25K dermoscopy images | 8-class multiclass | Faster turnaround, real multi-clinic provenance — natural non-IID story |

Skip: MedMNIST (toy), MIMIC-CXR (PhysioNet credentialing eats 2 weeks).

---

## 3. Architecture + baselines

- **Backbone**: ResNet-50 (ResNet-110 from the paper is too heavy at 224×224).
- **Split points**: shallow / middle / deep, same as paper.
- **Aux head**: 14 sigmoid heads + BCE for CheXpert (multi-label).
- **Baselines**: Centralized (upper bound) · CSL · DSL · ★ DSL-Weighted · FedAvg.
- Optional: SplitFed [Thapa 2022] if time permits.

---

## 4. Metrics

- Mean AUC-ROC + per-class AUC
- **Rare-class AUC gap** (Fracture, Pleural Other, Lung Lesion) — the novelty signal
- Communication MB / epoch
- Peak GPU memory MB
- Bandwidth-budget plot: AUC vs. cumulative MB sent (the clinical chart)

---

## 5. 8-Week Timeline

| Week | Work |
|---|---|
| **1** | PyTorch env. Reproduce paper's CIFAR-10 numbers for CSL + DSL (3 splits × N=1). Sign CheXpert DUA + ISIC accounts in parallel. |
| **2** | Reproduce N=5, N=10. Confirm comm + memory claims. Download CheXpert, build dataloader. |
| **3** | Port to ResNet-50 + multi-label BCE. Train centralized + CSL on CheXpert → reference AUCs. |
| **4** | Train DSL on CheXpert (3 splits × N=1). Plot convergence + comm + memory vs. CSL. Per-class AUC table → **identify the rare-class gap**. |
| **5** | Implement Dirichlet(α=0.1) non-IID partitioning across N=5 hospitals. Run DSL vs. CSL non-IID. Implement class-weighted aux loss. Ablation. |
| **6** | Apply pipeline to ISIC. Confirm findings hold. Add FedAvg baseline. |
| **7** | All figures: convergence, comm/mem bars, per-class AUC heatmap, bandwidth-budget plot. Failure-mode analysis. |
| **8** | Draft 8-page paper. Mentor review. Polish. |

STS deadline is mid-November. Hitting "draft done" by mid-August leaves
September–October for revisions, video, essays, recommendation letters.

---

## 6. Risks + mitigations

- **CheXpert is huge** → preprocessed 224×224 PNGs, cache as `.pt` after epoch 1.
- **ResNet-50 + N=10 + 3 splits may blow memory** → batch 64, gradient accumulation.
- **Multi-label BCE collapses to all-zero on rare classes** → use `pos_weight` from the start, even before the modification ablation.
- **Non-IID + small N diverges** → keep small λ > 0 on aux loss; becomes a discussion point.
- **STS judges want clear student ownership** → the class-weighted aux loss must be presented as the student's idea with motivation, math, ablation. Don't bury it.

## 7. Deliverables

1. Paper PDF (6–8 pages, IEEE or NeurIPS style)
2. GitHub repo (clean, README, reproducibility script)
3. 2-minute video (STS app requirement)
4. 250-word research summary essay

---

## 8. Paper Outline (Section-by-Section)

### Title

> *Decoupled Split Learning for Multi-Label Medical Imaging: Rare-Class
> Robustness via a Weighted Auxiliary Loss*

### Abstract

*Draft:*

> Split learning (SL) is a distributed training paradigm well suited to
> healthcare, where regulations prevent hospitals from sharing patient
> data and edge devices must operate under tight bandwidth budgets.
> Recent work on decoupled split learning (DSL) [Zihad et al., 2025]
> introduces a client-side auxiliary classifier that eliminates backward
> gradient transmission across the cut layer, halving communication and
> substantially reducing peak memory while preserving accuracy on
> CIFAR-10/100. However, DSL has not been evaluated on real medical
> data, which differs from natural-image benchmarks in three ways: it is
> multi-label (each chest X-ray carries several findings), severely
> class-imbalanced (rare diseases account for under 5% of labels), and
> non-IID across institutions (each hospital sees a different disease
> mix). In this paper, we present the first evaluation of DSL on
> multi-label medical imaging. On CheXpert (191K chest X-rays, 14
> conditions) and ISIC 2019 (25K dermoscopy images, 8 classes), we find
> that DSL matches conventional SL on mean AUC but suffers a [X.X%]
> AUC gap on rare classes. We propose a class-weighted auxiliary loss
> applied only to the client-side objective and show it closes [YY%] of
> this gap without affecting communication or memory cost, while
> preserving DSL's [50%] communication and [58%] memory savings. We
> also characterize DSL's behavior under Dirichlet-partitioned
> cross-hospital splits and find that the proposed weighting yields
> [Z.Z%] higher mean AUC at the most non-IID setting (α=0.1). Our
> results indicate that DSL is a practical option for privacy-preserving
> medical model training once the auxiliary objective is adapted to
> label imbalance.

### 1. Introduction (~1 page)

- Para 1 — clinical motivation (HIPAA, edge devices, real bandwidth limits).
- Para 2 — SL recap, then DSL: cite [Zihad 2025], explain in 2 sentences.
- Para 3 — but DSL only on CIFAR. Medical changes three things:
  1. **Multi-label** — single-softmax aux head won't work.
  2. **Class imbalance** — local aux loss may underweight rare classes.
  3. **Non-IID** — no-BP design may amplify drift across hospitals.
- Para 4 — ★ contribution bullets (below).
- Para 5 — paper roadmap.

**Contribution paragraph (drop in near end of intro):**

> In this paper, we make the following contributions:
>
> - **First medical-imaging evaluation of decoupled split learning.** We
>   adapt DSL to multi-label classification (replacing the single-softmax
>   auxiliary head with K independent sigmoid heads and per-label binary
>   cross-entropy) and benchmark it on CheXpert and ISIC 2019 against
>   conventional split learning, centralized training, and FedAvg. To our
>   knowledge, this is the first published evaluation of any BP-free
>   split-learning method on real medical data.
> - **Identification of a rare-class accuracy gap specific to DSL.** We
>   show that, although DSL matches conventional SL on aggregate AUC, it
>   consistently underperforms by [X.X%] on the rarest CheXpert findings
>   (Fracture, Pleural Other, Lung Lesion). We trace this to the structure
>   of the local auxiliary loss, which is dominated by majority-class
>   gradient signal because the client never sees a server-side
>   reweighting signal.
> - **A targeted fix: class-weighted auxiliary loss.** We propose weighting
>   only the client-side auxiliary loss by inverse class frequency,
>   leaving the server's global loss unchanged. The modification adds no
>   communication or memory overhead, requires no changes to the cut-layer
>   architecture, and closes [YY%] of the rare-class gap. We provide an
>   ablation across three weighting schemes (inverse frequency, effective
>   number of samples, log-scaled) and show inverse frequency performs
>   best on both datasets.
> - **Characterization of DSL under non-IID cross-hospital splits.** Using
>   Dirichlet partitioning with α ∈ {0.1, 0.5, 1.0, ∞}, we simulate five
>   hospitals with realistically divergent disease distributions and show
>   that DSL's accuracy gap relative to CSL widens as α decreases, but
>   that the weighted auxiliary loss recovers most of this drop. This is
>   the relevant regime for any real federated medical deployment.
> - **Bandwidth-budget framing for clinical deployment.** We translate the
>   communication-volume savings into wall-clock training time at
>   clinically realistic uplink speeds (1–50 Mbps) and show that DSL with
>   our modification reaches an AUC of [0.86] on CheXpert in [X hours]
>   over a 10 Mbps clinic uplink, compared to [2X hours] for conventional
>   SL — making on-site training tractable for under-resourced
>   facilities.

### 2. Related Work (~½ page)

- **Split learning**: Gupta & Raskar [1]; SplitFed [Thapa]; medical SL apps.
- **Beyond-BP training**: greedy layer-wise [Belilovsky]; synthetic gradients [Jaderberg]; DSL [Zihad].
- **FL on medical imaging**: brief — FedAvg, FedProx, the few CheXpert-via-FL papers. Position: "DSL has different tradeoffs than FL — lower client compute, unknown medical-data robustness."
- **What's NOT done**: "To our knowledge, no prior work evaluates DSL on multi-label medical data or under non-IID cross-hospital splits." Critical sentence — your originality claim.

### 3. Method (~1.5 pages)

**3.1 Decoupled split learning (recap)** — tight version of Zihad's formulation. One figure, the algorithm box. Cite heavily, don't pretend you invented it.

**3.2 ★ Multi-label auxiliary classifier** — replace softmax with K sigmoid heads + per-label BCE. Write out the new L_aux explicitly.

**3.3 ★ Class-weighted auxiliary loss** — see prose draft below.

**3.4 Non-IID partitioning** — Dirichlet(α=0.1) across N=5 simulated clients. Cite. Justify: real hospitals see Atelectasis at wildly different rates.

#### Prose draft of §3.3

> The vanilla multi-label auxiliary loss defined in Section 3.2,
>
> $$\mathcal{L}_{\text{aux}}(\tilde{\mathbf{y}}, \mathbf{y}) \;=\; -\frac{1}{B} \sum_{i=1}^{B} \sum_{k=1}^{K} \Big[ y_{i,k}\,\log\sigma(\tilde{y}_{i,k}) \;+\; (1 - y_{i,k})\,\log\bigl(1 - \sigma(\tilde{y}_{i,k})\bigr) \Big],$$
>
> treats every class equally and every sample equally. In medical
> imaging this is a poor inductive bias. On CheXpert, for example,
> positive prevalence ranges from 51.2% (Lung Opacity) down to 1.5%
> (Fracture). For a rare class $k$ with prevalence $p_k \ll 0.5$, the
> negative term $(1 - y_{i,k})\log(1 - \sigma(\tilde{y}_{i,k}))$ dominates
> the per-class gradient because $1 - y_{i,k} = 1$ for the vast majority
> of samples. The client's bottom model $M_b$ thus receives an auxiliary
> signal that rewards making $\tilde{y}_{i,k}$ small *unconditionally* —
> a representation that is locally optimal but discards the discriminative
> features needed to detect class $k$ when it does appear.
>
> In a centrally trained model, or in a conventional split-learning
> setup, the server's end-task loss $\mathcal{L}(\hat{\mathbf{y}},
> \mathbf{y})$ would correct this drift through backpropagation across
> the cut layer. DSL eliminates that backward channel by design (Section
> 3.1), so the rare-class signal must instead be injected into the
> client-side objective directly.
>
> **Per-class positive weighting.** We modify only the auxiliary loss:
>
> $$\mathcal{L}_{\text{aux}}^{w}(\tilde{\mathbf{y}}, \mathbf{y}) \;=\; -\frac{1}{B} \sum_{i=1}^{B} \sum_{k=1}^{K} \Big[ w_k \cdot y_{i,k}\,\log\sigma(\tilde{y}_{i,k}) \;+\; (1 - y_{i,k})\,\log\bigl(1 - \sigma(\tilde{y}_{i,k})\bigr) \Big],$$
>
> with $w_k = N_{\text{neg}}^{(k)} / N_{\text{pos}}^{(k)} = (1 - p_k) / p_k$.
> For Fracture ($p_k \approx 0.015$), this yields $w_k \approx 66$.
>
> **Why weight only the auxiliary loss.** The server already minimizes
> $\mathcal{L}$ using the true labels and converges to a well-calibrated
> decision boundary; up-weighting rare classes in the server's loss
> biases predictions toward false positives without addressing the root
> cause, which is the *representation* learned by $M_b$. The failure
> mode is client-side, so the fix is client-side.
>
> **Cost.** Computing $\{w_k\}_{k=1}^{K}$ is a single pass over training
> labels (negligible). The per-batch loss adds one elementwise
> multiplication of cost $O(BK)$ per forward call.
>
> **Ablation.** Three weighting schemes compared: (i) inverse-frequency
> $w_k = N / (K N_{\text{pos}}^{(k)})$ [Huang 2016]; (ii)
> effective-number-of-samples $w_k = (1-\beta)/(1-\beta^{N_{\text{pos}}^{(k)}})$
> with $\beta=(N-1)/N$ [Cui 2019]; (iii) per-class positive weighting
> (above, PyTorch `pos_weight`, [CheXNet 2017]). Section 4.4 reports (iii)
> wins on both datasets.

### 4. Experiments (~2.5 pages)

- **4.1 Setup** — datasets, architecture, hyperparameters, baselines, metrics.
- **4.2 RQ1 — Does DSL preserve mean AUC?** Table + convergence plot (mirror paper's Fig 4).
- **4.3 ★ RQ2 — Where does DSL lose?** Per-class AUC heatmap. **Central figure.** Quantify rare-class gap.
- **4.4 ★ RQ3 — Does class-weighted aux close the gap?** Ablation: vanilla DSL vs. DSL-Weighted. Try ≥2 weighting schemes.
- **4.5 RQ4 — Non-IID robustness.** Mean AUC vs. α plot.
- **4.6 Efficiency confirmation.** Comm + memory + training time (mirror paper's Fig 5/6/7).
- **4.7 ISIC validation.** Confirm findings hold on a second medical domain.

### 5. Discussion (~½ page)

- **Bandwidth-budget plot** — AUC vs. cumulative MB sent. Translate to clinical numbers.
- **Limitations** — simulated non-IID isn't real multi-hospital; ResNet-50 isn't SOTA on CheXpert; rare-class AUC noisy below 1000 positives.
- **Why DSL still loses on the rarest classes** — hypothesize the cut layer doesn't see enough rare-class signal. Future work: sample-balanced micro-batches.

### 6. Conclusion (~¼ page)

- Restate contributions in past tense.
- One sentence broader implication.
- One sentence future work (real federated deployment; medical time series).

### References

- DSL paper, Gupta & Raskar, SplitFed, CheXpert paper, ISIC paper, FedAvg, focal loss, class-balanced loss, Dirichlet non-IID FL papers. ~20 refs.

---

## 9. Writing-craft notes for the student

- **The contribution paragraph and §4.3 rare-class finding must be the most polished writing.** Most readers (and STS judges) read abstract + intro + one figure carefully. Rest is skimmed.
- **Cite [Zihad 2025] generously.** Be explicit about what's reused vs. new. STS judges have seen "I downloaded code and changed the dataset" passed off as original work — explicit attribution is a sign of integrity.
- **The equations are §3.3's spine.** Reviewers read math more carefully than prose. Every symbol defined once, notation matching §3.1.
- **"To our knowledge, this is the first" needs verification.** Before submission: Google Scholar + arXiv search for "decoupled split learning medical", "split learning chest X-ray", "BP-free federated medical". If something close exists, narrow the scope but keep the claim honest.
- **Bracketed numbers are commitments to your future self.** When experiments come back differently, update the abstract and contributions. Don't leave stale claims contradicting tables.

---

## 10. Open questions / decisions to revisit

- Which lab GPU is the student using? Confirm with the Montclair mentor.
- Who signs the CheXpert DUA (student is under 18 → the mentor)?
- Are there pre-existing CheXpert / ISIC pipelines in the lab? Reuse if so.
- Final dataset combo: is CheXpert + ISIC enough, or add a tabular EHR or time-series dataset for a third modality?
- Figure captions + table headers — draft early so experiments stay disciplined.

---

*Next session: pick up from §8 outline. Likely tasks — draft figure
captions / table headers, then start week-1 reproduction once compute
is available.*
