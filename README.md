# Fraud Detection using Graph Neural Networks

**IIT Kanpur Winter Project 2025–26** | Group 6

> Rounak Mandal (240885) · Sahaj Bindal (240900) · Poonam Gupta (240751)

---

## Why Graphs? The Core Idea

Financial fraud is almost never a solo act. Money laundering, transaction layering, and illicit fund transfers all involve **coordinated networks** of transactions designed to obscure where money came from and where it's going. Common techniques include:

- **Smurfing** — breaking large sums into many small transactions to avoid detection thresholds
- **Peeling chains** — rapidly hopping funds through intermediary wallets
- **Transaction layering** — creating deliberately complex paths to confuse investigators

Traditional fraud detection systems — rule engines, Random Forests, XGBoost — look at each transaction in isolation. They assume transactions are independent of each other (the I.I.D. assumption), which makes them structurally blind to these coordinated patterns. They also require expensive manual feature engineering to capture any relational signal, and even then they can't see connections more than one degree away.

Graph Neural Networks take a completely different approach. Instead of analyzing a transaction on its own, a GNN looks at *where that transaction sits within the broader network* — who it's connected to, what those connections look like, and what patterns emerge across multiple hops. This is exactly the kind of relational reasoning that fraud detection needs.

---

## The Dataset: Elliptic Bitcoin Dataset

We use the **Elliptic Bitcoin Dataset** — the standard academic benchmark for graph-based financial anomaly detection. It maps real Bitcoin blockchain activity into a transaction graph.

| Property | Value |
|----------|-------|
| Nodes (transactions) | 203,769 |
| Edges (payment flows) | 234,355 directed |
| Time steps | 49 (each ~2 weeks of activity) |
| Features per node | 166 |
| Illicit transactions | ~4,545 (~2%) |
| Licit transactions | ~42,019 (~21%) |
| Unknown/unlabelled | ~157,205 (~77%) |

Each node has **166 features** split into two groups:
- **94 local features** — transaction-level attributes like timestamp, fee, volume, and number of inputs/outputs
- **72 aggregated structural features** — manually engineered neighbourhood statistics (mean/std of neighbour transaction values, etc.)

The severe class imbalance — only 2% of labelled nodes are illicit — means a naive model that always predicts "licit" achieves 95% accuracy while catching zero fraud. This is why we focus on **Macro F1** and **PR-AUC** as our primary metrics, not accuracy.

---

## Repository Structure

```
fraud-Detection-using-GNN/
├── gnn-eda.ipynb                    # Exploratory data analysis
├── GNN_Experimentation1.ipynb       # First round: baseline GNN vs XGBoost
├── GNN_Experimentation2.ipynb       # Second round: improved methodology, fixed bugs
├── GNN_Final_Code.ipynb             # Final polished pipeline with all models
├── GNN Experimentation report .pdf  # Detailed technical report on experiments
└── GNN_Group6_FinalReport.pdf       # Final project report
```

---

## How We Approached It

### Step 1 — Exploratory Data Analysis (`gnn-eda.ipynb`)

Before touching a model, we studied the data. Key findings:
- The graph is highly sparse with a heavy-tailed degree distribution — a few hub nodes connect to many others, typical of real financial networks
- Illicit activity appears in **bursts** across timesteps, not uniformly distributed — fraud is time-dependent
- Features are already numeric and normalized, reducing preprocessing complexity
- Some features are highly correlated (redundant), while others carry unique signals

We also visualized subgraphs around illicit nodes — red nodes (illicit) tend to be surrounded by unknown/gray nodes, which reflects the real-world challenge of semi-supervised fraud detection.

### Step 2 — Feature Engineering

On top of the 166 raw features, we added:
- **Log-degree** and **log-PageRank** — Bitcoin transaction graphs follow power-law distributions, so raw values have extreme outliers that destabilize gradient descent. Log transforms compress the range while preserving relative ordering.
- **Centrality measures** — how important a node is within the graph structure
- **Temporal patterns** — capturing how transaction behaviour changes over time

**Key design decision:** GNN models receive only the 94 local features. The graph structure allows GNNs to *learn* neighbourhood statistics automatically through message passing — making the 72 pre-engineered structural features redundant. XGBoost gets all 166 features since it has no graph awareness and needs the full picture. This lets us directly measure how much the graph structure contributes.

### Step 3 — Data Splitting Without Leakage

We cannot split randomly here. Bitcoin transactions are ordered in time, and a model trained on future data to predict past fraud would be useless in the real world.

We split **chronologically** using the timestep feature:

| Split | Timesteps | Labelled Nodes |
|-------|-----------|----------------|
| Training | 1 – 34 | 29,894 |
| Validation | 35 – 38 | 4,303 |
| Test | 39 – 49 | 12,367 |

Unlabelled nodes (77% of the graph) are excluded from train/val/test but **kept in the graph structure** for message passing. This is the standard semi-supervised GNN setup — those unlabelled nodes still pass information to their labelled neighbours.

Feature scaling is also done carefully: the `StandardScaler` is fitted **exclusively on training nodes**, then applied to validation and test nodes separately. Fitting on the full dataset before splitting would contaminate the training process with future information.

---

## Model Architectures

### ResGCN — Graph Convolutional Network with Residual Connections

Standard GCNs suffer from two problems at depth: vanishing gradients and over-smoothing (where all node embeddings converge to the same vector after many hops). ResGCN fixes this with **skip connections** borrowed from ResNet in computer vision.

The architecture:
- Input projection: `Linear(94 → 512)` to lift raw features into hidden space
- Two GCN layers with BatchNorm, ReLU, Dropout(0.3), and a residual add after each
- Output GCN layer producing licit/illicit logits

ResGCN uses **undirected edges** because GCNConv's normalisation is based on a symmetric graph Laplacian — directed edges would produce incorrect row-normalisation.

### ImprovedGAT — Graph Attention Network

Standard GAT treats all neighbours as potentially equal and learns attention weights to differentiate them. ImprovedGAT adds an **input projection layer** and **residual connections** — improvements that turned out to be critical for preventing gradient stagnation observed in early training runs.

The 4-head attention mechanism lets the model simultaneously learn four different "views" of neighbourhood importance. For fraud, this is valuable: some neighbours are strong indicators of illicit activity while others are irrelevant noise.

Architecture: `Input Proj → GAT(heads=4) → BN → ELU → GAT(heads=4) → BN → ELU → GAT(heads=1)`

### GraphSAGE Ensemble — Inductive Graph Learning

GraphSAGE uses **max-pooling aggregation** to summarise neighbourhood information. Unlike GCN, it is inherently **inductive** — it can generalise to nodes not seen during training, making it suitable for real-time screening of new transactions.

We train **three independent GraphSAGE models** with different random seeds (0, 1, 2) and average their predicted probabilities before applying the decision threshold. Different seeds produce different weight initialisations and training dynamics — averaging over them reduces variance so that borderline cases aren't dominated by one model's uncertainty.

### XGBoost Baseline

A Gradient Boosted Tree model (500 estimators, max_depth=8, learning_rate=0.05) trained on all 166 features without any graph structure. This baseline exists for one purpose: to quantify the value-add of using the graph. If GNNs don't outperform XGBoost, the graph isn't helping.

---

## Handling Class Imbalance

With only 2% of labelled nodes being illicit, standard cross-entropy loss causes the model to ignore the minority class. We apply two layers of protection:

**Inverse-frequency class weights**
```
weight_licit   = n_total / (2 × n_licit)     →  ~0.565
weight_illicit = n_total / (2 × n_illicit)   →  ~4.317
```
The illicit class receives ~7.6× more weight during backpropagation.

**Focal Loss** (Lin et al., 2017 — originally developed for object detection)

Focal Loss adds a modulating factor `(1 - p_t)^γ` to each example's loss:
- Easy, correctly-classified examples: `p_t ≈ 1` → loss contribution is down-weighted
- Hard, misclassified examples: `p_t ≈ 0` → loss contribution is preserved

This forces the model to spend its capacity on the hard cases — which in fraud detection are almost always the rare illicit transactions. We tuned `γ = 2.0` via grid search.

**Threshold optimisation**

Rather than using 0.5 as the decision boundary, we search the range [0.1, 0.9] and pick the threshold that maximises Macro F1 on the validation set. With severe class imbalance, the optimal boundary is almost always above 0.5.

---

## Training Details

- **Optimiser:** AdamW with weight decay (L2 regularisation)
- **Scheduler:** ReduceLROnPlateau — halves the learning rate when validation Macro F1 stops improving for 15–20 steps. Enables coarse exploration early, fine-grained refinement later.
- **Early stopping:** Training halts when validation performance plateaus. The patience counter resets on any improvement — a subtle but important fix over a naive `epoch - best_epoch` implementation that could stop training prematurely.
- **Gradient clipping:** `max_norm=1.0` prevents exploding gradients on large irregular graphs where high-degree hub nodes can cause instability.
- **Best model checkpointing:** At each evaluation step, if validation Macro F1 improves, model weights are saved. Training always returns the best-performing checkpoint, not the final epoch.

---

## Results

All models evaluated on the held-out test set (timesteps 39–49):

| Model | Macro F1 | PR-AUC | Recall (illicit) | FPR | Threshold |
|-------|----------|--------|------------------|-----|-----------|
| ResGCN | 0.7200 | 0.3681 | 0.5202 | 0.0411 | 0.504 |
| ImprovedGAT | 0.7112 | 0.4306 | 0.4338 | 0.0293 | 0.536 |
| GraphSAGE Ensemble | 0.7607 | 0.4239 | 0.5119 | 0.0221 | 0.601 |
| **XGBoost** | **0.8699** | **0.7073** | **0.6206** | **0.0018** | 0.835 |

### What the numbers tell us

**XGBoost wins on overall metrics.** This is consistent with published literature on the Elliptic dataset — when 72 pre-engineered structural features already encode network structure, tree-based models that can directly exploit that rich tabular input are hard to beat.

**GNNs get a fairer fight than the numbers suggest.** GNN models received only 94 local features while XGBoost got all 166. Despite that, GNNs achieved competitive recall on illicit transactions, which shows that message passing is genuinely learning structural information from the graph — it's not just the features doing the work.

**There is a fundamental precision-recall tradeoff.** XGBoost is extremely precise (FPR = 0.0018 — almost no false alarms) but GNNs catch more illicit transactions at the cost of more false positives. In a real AML deployment, missing an illicit transaction carries far greater cost than a false alarm. A hybrid system — GNN for sensitivity, XGBoost for precision — could offer the best of both.

**GraphSAGE Ensemble is the most stable GNN.** Averaging probabilities across three independently-trained models smooths out variance from random initialisation, making it the most reliable of the three GNN architectures.

---

## Key Design Decisions & Bug Fixes

The second experimentation notebook documents several important fixes made over an initial baseline:

| # | Location | Fix |
|---|----------|-----|
| 1 | Graph build | Created a separate undirected edge index for GCN-family models |
| 2 | Features | Used all 165 original features, not just the 94 local ones (for XGBoost) |
| 3 | Scaling | StandardScaler fitted only on training rows — prevents data leakage |
| 4 | Data objects | GCN gets undirected data; GraphSAGE and GAT keep directed data |
| 5 | Class weights | Inverse-frequency formula, not normalised 1/n |
| 6 | Metrics | All evaluations use Macro F1, not binary F1, for fair class comparison |
| 7 | Early stopping | Proper patience counter that resets on improvement |
| 8 | XGBoost API | `early_stopping_rounds` moved to constructor (required in XGBoost ≥ 2.0) |
| 9 | XGBoost features | Uses unscaled features — tree models are invariant to feature scaling |

---

## Libraries

| Library | Role |
|---------|------|
| `torch` (PyTorch) | Model definition, training, GPU acceleration |
| `torch_geometric` | Graph data structures, GNN layers (GCNConv, SAGEConv, GATConv) |
| `xgboost` | Gradient boosted tree baseline |
| `scikit-learn` | StandardScaler, F1 score, PR-AUC, confusion matrix |
| `pandas` / `numpy` | Data loading and manipulation |
| `networkx` | Graph construction and subgraph visualisation |
| `matplotlib` | Plotting training curves and subgraphs |

---

## Future Directions

**Temporal GNNs (e.g., EvolveGCN)** — Rather than treating the 49 timesteps as a static graph, model the graph as it evolves over time. This would let the model detect when fraud patterns shift — concept drift is a real problem in financial AML.

**Heterogeneous graphs** — The current graph only has one type of node (transaction). Adding wallet addresses, exchanges, and mixer services as different node types would let the model distinguish between different kinds of financial entities.

**GNNExplainer** — For real-world compliance, investigators need to know *why* a transaction was flagged. GNNExplainer generates per-prediction subgraph explanations showing which specific transactions and features drove each illicit classification.

**Mini-batch training** — Full-batch GNN training on 203,769 nodes is memory-intensive. Moving to GraphSAGE neighbourhood sampling in mini-batches would enable real-time inference on streaming transaction data.

**Semi-supervised label propagation** — 77% of nodes are unlabelled. Pseudo-labelling or label propagation could turn that large unlabelled pool into a training signal, potentially improving GNN performance substantially.

---

## References

1. Weber, M., et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* AAAI Workshop.
2. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
3. Hamilton, W., Ying, Z., & Leskovec, J. (2017). *Inductive Representation Learning on Large Graphs.* NeurIPS.
4. Velickovic, P., et al. (2018). *Graph Attention Networks.* ICLR.
5. Lin, T. Y., et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
6. Ying, R., et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks.* NeurIPS.
7. Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop.
