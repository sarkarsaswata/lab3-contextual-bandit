# Contextual Bandit-Based News Recommendation System  
Lab 3 – Contextual Multi-Armed Bandits

---

## Overview

In this lab, I implemented a contextual multi-armed bandit system for personalized news recommendation.

The idea is to combine:

1. Supervised learning (to predict user context), and  
2. Reinforcement learning (to learn the best news category per context).

Each user is classified into one of three contexts:

- User1  
- User2  
- User3  

For each context, a 4-arm bandit is trained over the categories:

- Entertainment  
- Education  
- Tech  
- Crime  

This results in a total of 12 arms (3 contexts × 4 categories).  
Rewards are generated strictly using the provided `rlcmab_sampler`.

---

## 1. User Context Classification

I trained a Logistic Regression model to classify users into User1, User2, or User3.

Preprocessing steps:
- Dropped ID-like columns
- Median imputation for numeric features
- One-hot encoding for categorical features
- 80–20 stratified train-validation split

### Validation Performance

Validation Accuracy: **0.73**

| Class  | Precision | Recall | F1-score | Support |
|--------|----------|--------|----------|---------|
| User1  | 0.74     | 0.84   | 0.79     | 142     |
| User2  | 0.77     | 0.85   | 0.81     | 142     |
| User3  | 0.63     | 0.46   | 0.53     | 116     |

### Observations

- The classifier performs well for User1 and User2.
- User3 has significantly lower recall (0.46), meaning it is harder to correctly identify.
- This suggests feature overlap between User3 and other classes.
- Overall accuracy of 0.73 indicates reasonably strong context separation.

---

## 2. Contextual Bandit Setup

Once a user’s context is predicted, a separate 4-arm bandit is used for that context.

Arm mapping (as required):

- User1 → arms 0–3  
- User2 → arms 4–7  
- User3 → arms 8–11  

Within each context:

0 → Entertainment  
1 → Education  
2 → Tech  
3 → Crime  

Each strategy was simulated for:

T = 10,000 time steps per context.

---

## 3. Bandit Strategies & Results

### User1

#### Epsilon-Greedy

| ε     | Final Avg Reward |
|-------|------------------|
| 0.01  | 7.0515 |
| 0.05  | 6.7190 |
| 0.10  | 6.2348 |

Best ε = **0.01**

#### UCB

| C   | Final Avg Reward |
|-----|------------------|
| 1.0 | 7.1362 |
| 0.5 | 7.1291 |
| 2.0 | 7.1162 |

Best C = **1.0**

#### Softmax (τ = 1.0)

Final Avg Reward = **7.1188**

**Observation:**  
UCB slightly outperformed both epsilon-greedy and softmax.  
The differences are small but consistent, with C = 1.0 performing best.

---

### User2

#### Epsilon-Greedy

| ε     | Final Avg Reward |
|-------|------------------|
| 0.01  | 5.0684 |
| 0.05  | 4.7599 |
| 0.10  | 4.4961 |

Best ε = **0.01**

#### UCB

| C   | Final Avg Reward |
|-----|------------------|
| 2.0 | 5.1319 |
| 0.5 | 5.1285 |
| 1.0 | 5.1211 |

Best C = **2.0**

#### Softmax (τ = 1.0)

Final Avg Reward = **5.1028**

**Observation:**  
UCB again achieved the highest reward.  
Here, a larger C (2.0) performed slightly better, indicating that stronger exploration helped in this context.

---

### User3

#### Epsilon-Greedy

| ε     | Final Avg Reward |
|-------|------------------|
| 0.05  | 6.3991 |
| 0.10  | 6.2312 |
| 0.01  | 5.8566 |

Best ε = **0.05**

#### UCB

| C   | Final Avg Reward |
|-----|------------------|
| 0.5 | 6.5725 |
| 1.0 | 6.5543 |
| 2.0 | 6.5411 |

Best C = **0.5**

#### Softmax (τ = 1.0)

Final Avg Reward = **6.3003**

**Observation:**  
User3 behaves differently compared to the other contexts.  
Here, ε = 0.05 performed best in epsilon-greedy, suggesting that slightly more exploration was necessary.  
However, UCB with C = 0.5 still achieved the highest overall reward.

---

## 4. Hyperparameter Summary

Best parameters per context:

Epsilon-Greedy:
- User1 → ε = 0.01  
- User2 → ε = 0.01  
- User3 → ε = 0.05  

UCB:
- User1 → C = 1.0  
- User2 → C = 2.0  
- User3 → C = 0.5  

Across all contexts, UCB consistently achieved the highest final average reward.

---

## 5. Overall Observations

1. UCB outperformed epsilon-greedy and softmax in all three contexts.
2. The optimal exploration parameter varies across user types.
3. User2 benefits from stronger exploration (higher C).
4. User3 required moderate exploration (ε = 0.05).
5. With T = 10,000, all strategies clearly converge.

These results confirm that:
- Contextual separation is meaningful.
- Different user types exhibit different reward structures.
- Adaptive exploration (UCB) is generally more robust than fixed exploration strategies.

---

## 6. Final Recommendation Policy

For deployment, I selected the best-performing UCB configuration per context.

For each test user:

1. Predict context using the trained classifier.
2. Select the arm with the highest learned Q-value.
3. Sample a real article from the corresponding category.
4. Output:
   - user_id  
   - predicted context  
   - recommended category  
   - headline  
   - link  
   - date  
   - authors  
   - short description  

Final output file:

lab3_recommendations_128.csv

---

## 7. How to reproduce the experiments


Important:
- Do not modify `rlcmab_sampler.py`
- Do not change arm indexing
- Ensure your roll number is correctly set inside the notebook

---

Start Jupyter:

Run all cells sequentially (Kernel → Restart & Run All).

The notebook performs the following steps automatically:

1. Loads and inspects datasets.
2. Trains the Logistic Regression classifier.
3. Prints validation accuracy and classification report.
4. Runs bandit simulations (T = 10,000) for:
   - Epsilon-Greedy (3 ε values)
   - UCB (3 C values)
   - Softmax (τ = 1.0)
5. Plots:
   - Average reward vs time (per context and strategy)
6. Prints:
   - Final average rewards
   - Best hyperparameters per context
7. Trains final policy using best UCB configuration.
8. Generates recommendations for `test_users.csv`.
9. Saves output file:


---


To reproduce the exact same numbers shown in this README:

- Ensure the random seed inside the notebook is fixed.
- Do not change the roll number used in `sampler(ROLL_NUMBER)`.
- Run all cells in order without re-running individual bandit sections separately.
- Avoid interrupting execution midway.

Since reward sampling depends on the roll number, changing it will produce different reward distributions.

---


After running all cells, you should observe:

- Validation accuracy ≈ 0.73
- Reward convergence curves for all contexts
- UCB performing best across contexts
- Final average rewards close to those reported above
- Generated recommendation CSV file

---

Notes on Randomness

- Bandit algorithms use stochastic exploration.
- Rewards are sampled from a stochastic simulator.
- With fixed seeds and unchanged roll number, results are reproducible.
- Minor floating-point differences may occur depending on system/OS.

---

If any output differs significantly:

- Verify arm mapping (0–3, 4–7, 8–11).
- Ensure `T = 10000`.
- Check that hyperparameters match the specified values.





