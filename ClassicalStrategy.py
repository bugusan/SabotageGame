import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Reproducibility ---
# Setting the seed ensures all random choices (defense, classical strategy, rule-based choices) are repeatable.
NP_RANDOM_SEED = 20251021
np.random.seed(NP_RANDOM_SEED)

# --- Custom Parameter ---
FONT_SCALE_FACTOR = 1.25  # Increase fonts by 25%

# Ensure output directory exists
OUT_DIR = "ClassicalStrategy_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# === Game Parameters (New Player Counts) ===
num_rounds = 100
num_bell_players = 2  # Bell Team (2Q)
num_w_players = 3     # W-state Team (3Q)
num_classical_2_players = 2 # Classical Team (2C) - Fixed 2-player baseline
# Variable-sized classical team (currently set to 3)
num_variable_classical_players = 3 # Classical Team (3C) 

# === Army Strategy (Defense Setup) ===
def generate_army_defense():
    paths = ['A', 'B']
    defended_path = np.random.choice(paths)
    return {path: "Strong" if path == defended_path else "None" for path in paths}

# === Classical Team Strategy ===
def classical_team_strategy(num_players):
    return np.random.choice(['A', 'B'], num_players)

# === Bell HAH (Hybrid Adaptive Heuristic Proxy for 2Q) ===
def bell_hah_strategy(army_defense, num_players=num_bell_players):
    """Hybrid Adaptive Heuristic (HAH) for 2Q: Rule-based, perfect coordination proxy."""
    first_choice = np.random.choice(['A', 'B'])
    second_choice = first_choice
    # The adaptive logic: if the random choice was bad, the team coordinates to mitigate the loss.
    if army_defense[first_choice] == "Strong":
        second_choice = 'A' if first_choice == 'B' else 'B'
    return [first_choice, second_choice]

# === W-state HAH (Hybrid Adaptive Heuristic Proxy for 3Q) ===
def w_hah_strategy(army_defense, num_players=num_w_players):
    """Hybrid Adaptive Heuristic (HAH) for 3Q: Rule-based, distributed, perfect coordination proxy."""
    first_choice = np.random.choice(['A', 'B'])
    
    # Initialize choices for N players
    team_choices = [first_choice] * num_players
    
    # The adaptive logic: if the initial choice was bad, the team perfectly flips N-1 players 
    # to maximize the utility, reflecting the theoretical ceiling of W-state coordination.
    if army_defense[first_choice] == "Strong":
        # Determine the opposite choice
        opposite_choice = 'A' if first_choice == 'B' else 'B'
        
        # N-1 players choose the opposite path (perfect coordination on opposite target)
        team_choices = [opposite_choice] * (num_players - 1)
        
        # Keep first player's original choice 
        team_choices.insert(0, first_choice) 
        
    return team_choices

# === Evaluate Effectiveness ===
def evaluate_effectiveness(choices, army_defense):
    return sum(1 if army_defense[c] == "None" else -1 for c in choices)

# === Run Simulation ===
# Scores for four groups
classical_2_scores, classical_variable_scores, quantum_w_scores_hah, quantum_bell_scores_hah = [], [], [], []
cumulative_classical_2, cumulative_classical_variable, cumulative_quantum_w_hah, cumulative_quantum_bell_hah = [0], [0], [0], [0]

for rnd in range(num_rounds):
    defense = generate_army_defense()

    # Classical (2 Players)
    c2_choice = classical_team_strategy(num_classical_2_players)
    c2_eff = evaluate_effectiveness(c2_choice, defense)
    classical_2_scores.append(c2_eff)
    cumulative_classical_2.append(cumulative_classical_2[-1] + c2_eff)

    # Classical (Variable Players - currently 3)
    c_variable_choice = classical_team_strategy(num_variable_classical_players)
    c_variable_eff = evaluate_effectiveness(c_variable_choice, defense)
    classical_variable_scores.append(c_variable_eff)
    cumulative_classical_variable.append(cumulative_classical_variable[-1] + c_variable_eff)
    
    # Bell HAH (2Q)
    bell_choice_hah = bell_hah_strategy(defense)
    bell_eff_hah = evaluate_effectiveness(bell_choice_hah, defense)
    quantum_bell_scores_hah.append(bell_eff_hah)
    cumulative_quantum_bell_hah.append(cumulative_quantum_bell_hah[-1] + bell_eff_hah)

    # W-state HAH (3Q)
    w_choice_hah = w_hah_strategy(defense)
    w_eff_hah = evaluate_effectiveness(w_choice_hah, defense)
    quantum_w_scores_hah.append(w_eff_hah)
    cumulative_quantum_w_hah.append(cumulative_quantum_w_hah[-1] + w_eff_hah)

# ==============================================================================
# 1. SAVE CSV DATA
# ==============================================================================
df_accumulated = pd.DataFrame({
    "Round": list(range(num_rounds + 1)),
    "Classical_2_Accumulated_Score": cumulative_classical_2,
    f"Classical_{num_variable_classical_players}_Accumulated_Score": cumulative_classical_variable,
    "Bell_Accumulated_Score_HAH": cumulative_quantum_bell_hah,
    "W_State_Accumulated_Score_HAH": cumulative_quantum_w_hah,
})
csv_path_accumulated = os.path.join(OUT_DIR, "Accumulated_Scores_RuleBased.csv") # Kept file name the same
df_accumulated.to_csv(csv_path_accumulated, index=False)
print(f"Saved accumulated score data to: {csv_path_accumulated}")

df_effectiveness = pd.DataFrame({
    "Round": list(range(num_rounds)),
    "Classical_2_Effectiveness_Score": classical_2_scores,
    f"Classical_{num_variable_classical_players}_Effectiveness_Score": classical_variable_scores,
    "Bell_Effectiveness_Score_HAH": quantum_bell_scores_hah,
    "W_State_Effectiveness_Score_HAH": quantum_w_scores_hah,
})
csv_path_effectiveness = os.path.join(OUT_DIR, "Effectiveness_Scores_RuleBased.csv") # Kept file name the same
df_effectiveness.to_csv(csv_path_effectiveness, index=False)
print(f"Saved effectiveness score data to: {csv_path_effectiveness}")

# ==============================================================================
# 2. PLOTTING WITH INCREASED FONT SIZE (25%)
# ==============================================================================
SMALL_SIZE = 10 * FONT_SCALE_FACTOR
MEDIUM_SIZE = 12 * FONT_SCALE_FACTOR
LARGE_SIZE = 14 * FONT_SCALE_FACTOR

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=LARGE_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=LARGE_SIZE)

# --- Accumulated Scores Plot ---
plt.figure(figsize=(10, 6))
plt.plot(range(num_rounds + 1), cumulative_classical_2,
          label=f"Classical Team ({num_classical_2_players} Players, 2C)", color='red', linestyle='--')
plt.plot(range(num_rounds + 1), cumulative_classical_variable,
          label=f"Classical Team ({num_variable_classical_players} Players, {num_variable_classical_players}C)", color='red', linestyle='-')
plt.plot(range(num_rounds + 1), cumulative_quantum_bell_hah,
          label=f"Bell (HAH, {num_bell_players} players, 2Q)", color='blue', linestyle='-')
plt.plot(range(num_rounds + 1), cumulative_quantum_w_hah,
          label=f"W-state (HAH, {num_w_players} players, {num_w_players}Q)", color='green', linestyle='-')
plt.xlabel("Round Number")
plt.ylabel("Accumulated Score")
plt.title("Accumulated Score Over Rounds (Hybrid Adaptive Heuristic vs Classical)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ClassicalStrategy_Accumulated_Score.png"), dpi=300) # Kept file name the same
# plt.show() # Disabled plt.show() for terminal execution

# --- Side-by-side Histogram Plot ---
mean_classical_2 = np.mean(classical_2_scores)
mean_classical_variable = np.mean(classical_variable_scores)
mean_bell_hah = np.mean(quantum_bell_scores_hah)
mean_w_hah = np.mean(quantum_w_scores_hah)

# Bin edges should cover the min (-3 for 3 players) to max (+3 for 3 players) range
max_score = max(num_variable_classical_players, num_w_players)
min_score = -max_score
bin_edges = np.arange(min_score - 1, max_score + 2) 
width = 0.20 # Adjust width for 4 groups

plt.figure(figsize=(12, 7))
plt.hist(classical_2_scores, bins=bin_edges + 0*width, alpha=0.7, color='red', hatch='//',
          label=f"Classical ({num_classical_2_players}C) (Mean: {mean_classical_2:.2f})",
          rwidth=width, align='left')
plt.hist(classical_variable_scores, bins=bin_edges + 1*width, alpha=0.8, color='red',
          label=f"Classical ({num_variable_classical_players}C) (Mean: {mean_classical_variable:.2f})",
          rwidth=width, align='left')
plt.hist(quantum_bell_scores_hah, bins=bin_edges + 2*width, alpha=0.8, color='blue',
          label=f"Bell (HAH, {num_bell_players}Q) (Mean: {mean_bell_hah:.2f})",
          rwidth=width, align='left')
plt.hist(quantum_w_scores_hah, bins=bin_edges + 3*width, alpha=0.8, color='green',
          label=f"W-state (HAH, {num_w_players}Q) (Mean: {mean_w_hah:.2f})",
          rwidth=width, align='left')

plt.xlabel("Effectiveness Score")
plt.ylabel("Frequency")
plt.title("Effectiveness Comparison: Classical vs Hybrid Adaptive Heuristic (HAH)")
# Center the ticks roughly between the 2nd and 3rd group for readability
plt.xticks(np.arange(min_score, max_score + 1) + 1.5*width)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ClassicalStrategy_Effectiveness_Comparison.png"), dpi=300) # Kept file name the same
# plt.show() # Disabled plt.show() for terminal execution