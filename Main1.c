import numpy as np
import pandas as pd

# Simulated incomplete dataset
# Let's assume a network with two binary variables: A (hidden), B (observed)
data = pd.DataFrame({
    'A': [0, 1, np.nan, 0, np.nan, 1, np.nan],
    'B': [1, 1, 0, 0, 1, 0, 1]
})

# Initialize parameters randomly
P_A = 0.5
P_B_given_A = [0.6, 0.3]  # P(B=1|A=0), P(B=1|A=1)

def e_step(data, P_A, P_B_given_A):
    responsibilities = []
    for i, row in data.iterrows():
        if np.isnan(row['A']):
            # Compute responsibilities using Bayes rule
            prob_A0 = P_A * (P_B_given_A[0] if row['B'] == 1 else 1 - P_B_given_A[0])
            prob_A1 = (1 - P_A) * (P_B_given_A[1] if row['B'] == 1 else 1 - P_B_given_A[1])
            total = prob_A0 + prob_A1
            responsibilities.append(prob_A0 / total)
        else:
            responsibilities.append(row['A'])
    return np.array(responsibilities)

def m_step(data, responsibilities):
    # Update P(A)
    P_A_new = np.mean(responsibilities)
    
    # Update P(B|A)
    B_given_A0 = []
    B_given_A1 = []
    for i, row in data.iterrows():
        r = responsibilities[i]
        if row['B'] == 1:
            B_given_A0.append(1 - r)
            B_given_A1.append(r)
        else:
            B_given_A0.append(0)
            B_given_A1.append(0)
    P_B_given_A0_new = np.sum(B_given_A0) / np.sum(1 - responsibilities)
    P_B_given_A1_new = np.sum(B_given_A1) / np.sum(responsibilities)
    
    return P_A_new, [P_B_given_A0_new, P_B_given_A1_new]

# EM iterations
for i in range(10):
    responsibilities = e_step(data, P_A, P_B_given_A)
    P_A, P_B_given_A = m_step(data, responsibilities)

print("Final parameters:")
print("P(A=0):", P_A)
print("P(B=1|A=0):", P_B_given_A[0])
print("P(B=1|A=1):", P_B_given_A[1])
