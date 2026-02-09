# Lab 3: Contextual Bandit-Based News Article Recommendation

**Student:** Sandeep Ram  
**Roll Number:** U20230083  
**Branch:** `Sandeep_U20230083`

---

## Overview

This project implements a **Contextual Multi-Armed Bandit (CMAB)** news recommendation system. Given a user with behavioral features, the system:

1. **Classifies** the user into one of 3 user categories (contexts)
2. **Selects** the optimal news category using a trained bandit policy
3. **Recommends** a specific article from that category

The environment is modeled with **3 contexts** (user types) × **4 arms** (news categories) = **12 total arms**, where rewards come from Bernoulli distributions provided by the `rlcmab_sampler` package.

---

## Repository Structure

```
├── lab3_results_U20230083.ipynb   # Main notebook with all code, results, and plots
├── master.ipynb                   # Original template (not modified)
├── rlcmab_sampler.py              # Reward sampling module (used as-is)
├── README.md                      # This report
├── assignment.pdf                 # Lab handout
└── data/
    ├── news_articles.csv          # 209,527 news articles across 25 categories
    ├── train_users.csv            # 2,000 users with labels (user_1/2/3)
    └── test_users.csv             # 2,000 users without labels
```

---

## Approach

### 1. Data Preprocessing (Section 5.1)

- **Missing values:** Dropped rows with NaN values (train: 2000→1302, test: 2000→1321, news: 209527→156859)
- **User encoding:** Encoded `user_1`, `user_2`, `user_3` labels to integers 0, 1, 2 using `LabelEncoder`
- **News filtering:** Filtered to only the 4 target categories (ENTERTAINMENT, EDUCATION, TECH, CRIME → 18,130 articles)
- **Feature selection:** Dropped non-numeric columns (`user_id`, `browser_version`, `region_code`, `subscriber`) to prepare 28 numeric features for classification

### 2. User Classification (Section 5.2)

- **Model:** Random Forest Classifier (100 trees, `random_state=42`)
- **Split:** 80% training (1,041 samples) / 20% validation (261 samples), stratified
- **Validation accuracy:** ~89%
- This classifier serves as the **context detector** — predicting which user type a new user belongs to

### 3. Contextual Bandit Algorithms (Section 5.3)

All three algorithms maintain per-arm Q-value estimates (12 arms total) and operate within the context of the predicted user type.

**Arm mapping** follows Table 1 from the assignment:
| Arms j | Context | Categories |
|--------|---------|------------|
| 0–3 | user_1 | Entertainment, Education, Tech, Crime |
| 4–7 | user_2 | Entertainment, Education, Tech, Crime |
| 8–11 | user_3 | Entertainment, Education, Tech, Crime |

#### Epsilon-Greedy
- With probability ε, explores randomly within the context; otherwise exploits the best-known arm
- Tested ε ∈ {0.01, 0.05, 0.1, 0.2}

#### UCB (Upper Confidence Bound)
- Selects the arm maximizing Q(a) + c·√(ln(t)/N(a)) within the context
- Untried arms are prioritized
- Tested c ∈ {0.5, 1.0, 2.0, 3.0}

#### SoftMax (Boltzmann)
- Selects arms with probability proportional to exp(Q(a)/τ)
- Fixed temperature τ = 1.0

### 4. Recommendation Engine (Section 5.4)

End-to-end pipeline applied to all test users:
1. Extract numeric features from a test user
2. Classify → predicted user category (context)
3. Look up the bandit's best arm for that context
4. Randomly sample an article from the selected news category

### 5. Evaluation (Section 5.5)

- **Classification report** with per-class precision, recall, and F1
- **RL simulation** run for T = 10,000 steps for each algorithm
- **4 plots** generated:
  - Overall average reward vs time (3 algorithms)
  - Per-context average reward vs time
  - ε-Greedy hyperparameter comparison
  - UCB hyperparameter comparison

---

## Key Results

### Simulation Performance (T = 10,000)

| Algorithm | Avg Reward |
|-----------|-----------|
| Epsilon-Greedy (ε=0.1) | ~0.645 |
| UCB (c=2.0) | ~0.627 |
| SoftMax (τ=1.0) | ~0.469 |

### Learned Optimal Arms

| Context | Best Category | Q-value |
|---------|--------------|---------|
| user_1 | ENTERTAINMENT | ~0.55 |
| user_2 | EDUCATION | ~0.67 |
| user_3 | TECH | ~0.81 |

All three algorithms converge to the same optimal arm per context, confirming the underlying reward structure.

### Hyperparameter Analysis

**Epsilon-Greedy:**
- ε = 0.01: Under-explores, lower overall reward (~0.60)
- ε = 0.05–0.10: Best performance range (~0.65)
- ε = 0.20: Over-explores, reward drops (~0.63)

**UCB:**
- c = 0.5: Best performance (~0.67), exploits quickly
- c = 1.0: Close second (~0.65)
- c = 2.0–3.0: Over-exploration hurts convergence (~0.59–0.63)

---

## How to Reproduce

1. **Clone and checkout the branch:**
   ```bash
   git clone https://github.com/sandeepram7/lab3-contextual-bandit.git
   cd lab3-contextual-bandit
   git checkout Sandeep_U20230083
   ```

2. **Install dependencies** (Python ≥ 3.12):
   ```bash
   pip install numpy pandas matplotlib scikit-learn rlcmab-sampler
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook lab3_results_U20230083.ipynb
   ```
   Execute all cells top to bottom. The notebook is self-contained and produces all results and plots.

---

## Dependencies

- Python ≥ 3.12
- numpy
- pandas
- matplotlib
- scikit-learn
- rlcmab-sampler (provided package)
