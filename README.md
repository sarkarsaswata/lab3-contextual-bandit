# Lab 3: Contextual Bandit-Based News Article Recommendation

**Course**: Reinforcement Learning Fundamentals  
**Student**: Sher Partap Singh  
**Roll Number**: U20230081  
**Branch**: `sher_U20230081`

---

## 📋 Overview

This project implements a **Contextual Multi-Armed Bandit (CMAB)** system for personalized news article recommendations. The system:
1. Classifies users into categories (User1, User2, User3) based on behavioral features
2. Uses bandit algorithms to learn optimal news category recommendations per user type
3. Samples articles from the selected category to maximize engagement

---

## 🏗️ Architecture

```
User Features → [Classifier] → User Context → [Bandit Policy] → News Category → [Sampler] → Article
```

**Problem Formulation:**
- **Contexts**: 3 user types (User1, User2, User3)
- **Arms**: 4 news categories (Entertainment, Education, Tech, Crime) per context
- **Total Arms**: 12 (3 contexts × 4 categories)

---

## 📊 Results Summary

### User Classification
| Metric | Value |
|--------|-------|
| Model | RandomForestClassifier |
| Train/Val Split | 80/20 |
| Validation Accuracy | **89.75%** |

**Top Features**: `region_code`, `session_duration`, `browsing_depth`, `scroll_activity`, `time_on_site`

### Bandit Algorithm Performance (T=10,000 steps)

| Algorithm | Best Hyperparameter | Average Reward |
|-----------|---------------------|----------------|
| **UCB** | C=0.5 | **7.13** |
| SoftMax | τ=1.0 | 7.02 |
| ε-Greedy | ε=0.01 | 6.62 |

---

## 🔬 Algorithm Analysis

### Epsilon-Greedy
- Simple exploration-exploitation strategy
- **ε=0.01**: Fast convergence, risk of missing optimal arms
- **ε=0.1**: Good balance for most scenarios
- **ε=0.3**: Extensive exploration, slower learning

### Upper Confidence Bound (UCB)
- Systematic uncertainty-based exploration
- **C=0.5**: More exploitation, faster convergence
- **C=1.0**: Balanced approach
- **C=2.0**: More exploration, better long-term performance
- **Best performer** overall without manual tuning

### SoftMax
- Probabilistic selection via Boltzmann distribution
- **τ=1.0**: Fixed temperature parameter
- Smooth exploration-exploitation transition
- Sensitive to Q-value scales

---

## 📈 Key Insights

### Hyperparameter Sensitivity
| Parameter | Low Value Effect | High Value Effect |
|-----------|------------------|-------------------|
| ε (Epsilon-Greedy) | Fast convergence, may miss optimal | More exploration, slower learning |
| C (UCB) | Exploitation-focused | Exploration-focused |
| τ (SoftMax) | Greedy behavior | Uniform random |

### Strengths
✅ Adapts recommendations based on user context  
✅ Online learning enables continuous improvement  
✅ Computationally efficient algorithms  
✅ No large historical datasets required  

### Limitations
⚠️ Assumes stationary reward distributions  
⚠️ Classification accuracy impacts bandit performance  
⚠️ Cold-start problem for new users/categories  

---

## 🚀 Production Recommendations

1. **Use UCB** for robust performance without extensive tuning
2. **A/B test** hyperparameters in production
3. **Ensemble methods** for robustness
4. **Periodic retraining** as user preferences evolve

---

## 📁 Repository Structure

```
├── lab3_results_U20230081.ipynb  # Main notebook with all code & results
├── README.md                      # Project report (this file)
├── data/
│   ├── news_articles.csv         # News articles dataset
│   ├── train_users.csv           # Training user data (with labels)
│   └── test_users.csv            # Test user data (no labels)
├── assignment.pdf                # Lab assignment specification
└── Goal.md                       # Assignment requirements reference
```

---

## 🛠️ How to Run

```bash
# Install dependencies
pip install rlcmab-sampler numpy pandas matplotlib scikit-learn

# Run the notebook
jupyter notebook lab3_results_U20230081.ipynb
```

Execute all cells top-to-bottom. The notebook includes:
- Data loading & preprocessing
- User classification model training
- Bandit algorithm implementations
- RL simulations (T=10,000 steps)
- Visualization & analysis plots

---

## 📌 Conclusion

The **UCB algorithm with C=0.5** achieved the best performance with an average reward of **7.13** over 10,000 steps. The contextual bandit framework successfully balances exploration and exploitation for personalized news recommendations, with algorithm selection depending on computational resources, desired exploration-exploitation trade-off, and real-time performance requirements.
