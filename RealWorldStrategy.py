import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
# NOTE: Using Aer simulator, which is robust for this type of calculation.

# === Reproducibility and Style Parameters ===
# 1. Seeds for reproducibility (critical fix)
NP_RANDOM_SEED = 20251021
SHOTS = 1024 # Increased shots for statistical stability
SIMULATOR_SEED = 20251021
TRANSPILE_SEED = 20251021
np.random.seed(NP_RANDOM_SEED)

# 2. Font scaling (25% increase)
FONT_SCALE_FACTOR = 1.25

# Ensure output directory exists
OUT_DIR = "RealWorldStrategy_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# === Game Parameters (Dynamic Player Counts) ===
num_rounds = 100
num_bell_players = 2 
num_classical_2_players = 2 
num_w_players = 3  # W-State 
num_variable_classical_players = 3 

simulator = Aer.get_backend("aer_simulator")


def choices_from_bitstring(bitstr, num_players):
    """
    Maps a single measured bitstring to sabotage choices.
    Convention: '1' -> A (Target 1), '0' -> B (Target 0).
    Bitstring is reversed to map q[i] to Player i.
    """
    # Reverse bitstring to map c[0] to player 0 (right-most bit)
    bits = bitstr[::-1]
    # Ensure result has correct number of choices
    return ['A' if b == '1' else 'B' for b in bits[:num_players]]


def generate_bell_state():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    # Transpile with a fixed seed
    return transpile(qc, simulator, optimization_level=1, seed_transpiler=TRANSPILE_SEED)

def generate_w_state(num_qubits):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.x(0)  # Start from |10...0>
    
    # === GENERALIZED R-Y/CX CASCADE FOR W-STATE PREPARATION ===
    for i in range(num_qubits - 1):
        # Angle formula: theta_k = 2 * arccos(sqrt((N - k - 1) / (N - k)))
        theta = 2 * np.arccos(np.sqrt((num_qubits - i - 1) / (num_qubits - i)))
        
        # Apply the rotation and the controlled-X gate
        qc.ry(theta, i)
        qc.cx(i, i + 1)
        
    qc.measure(range(num_qubits), range(num_qubits))
    # Transpile with a fixed seed
    return transpile(qc, simulator, optimization_level=1, seed_transpiler=TRANSPILE_SEED)


def run_quantum_circuit(circuit, num_players):
    """Runs circuit, samples one bitstring from the memory (1024 shots)."""
    job = simulator.run(circuit, shots=SHOTS, seed_simulator=SIMULATOR_SEED, memory=True)
    result = job.result()
    
    # Get all measured bitstrings
    memory = result.get_memory() 
    
    # Sample one bitstring randomly from the 1024 outcomes for this round
    sampled_bitstring = np.random.choice(memory)

    # Convert the sampled bitstring to A/B choices
    return choices_from_bitstring(sampled_bitstring, num_players)

# === Strategies ===
def generate_army_defense():
    paths = ['A', 'B']
    defended = np.random.choice(paths)
    return {path: ("Strong" if path == defended else "None") for path in paths}

def classical_team_strategy(num_players):
    # This function is dynamic and correctly uses 'num_players'
    return np.random.choice(['A', 'B'], num_players)

def bell_state_strategy():
    # Pure measurement result, no classical adaptation
    return run_quantum_circuit(generate_bell_state(), num_bell_players)

def w_state_strategy():
    # Pure measurement result, no classical adaptation
    return run_quantum_circuit(generate_w_state(num_w_players), num_w_players)

def evaluate_effectiveness(choices, army_defense):
    return sum(1 if army_defense[c] == "None" else -1 for c in choices)

# === Simulation Loop ===
classical_2_scores, classical_variable_scores, quantum_w_scores, quantum_bell_scores = [], [], [], []
accum_classical_2, accum_classical_variable, accum_quantum_w, accum_quantum_bell = [0], [0], [0], [0]

for rnd in range(num_rounds):
    defense = generate_army_defense()

    # Classical (2 Players, 2C)
    c2_choice = classical_team_strategy(num_classical_2_players)
    c2_eff = evaluate_effectiveness(c2_choice, defense)
    classical_2_scores.append(c2_eff)
    accum_classical_2.append(accum_classical_2[-1] + c2_eff)

    # Classical (Variable Players, e.g., 5C)
    c_variable_choice = classical_team_strategy(num_variable_classical_players)
    c_variable_eff = evaluate_effectiveness(c_variable_choice, defense)
    classical_variable_scores.append(c_variable_eff)
    accum_classical_variable.append(accum_classical_variable[-1] + c_variable_eff)

    # Bell (2Q)
    bell = bell_state_strategy()
    beff = evaluate_effectiveness(bell, defense)
    quantum_bell_scores.append(beff)
    accum_quantum_bell.append(accum_quantum_bell[-1] + beff)

    # W-state (N Qubits, e.g., 5Q)
    w = w_state_strategy()
    weff = evaluate_effectiveness(w, defense)
    quantum_w_scores.append(weff)
    accum_quantum_w.append(accum_quantum_w[-1] + weff)

# === Statistics ===
mean_classical_2 = float(np.mean(classical_2_scores))
mean_classical_variable = float(np.mean(classical_variable_scores))
mean_bell = float(np.mean(quantum_bell_scores))
mean_w = float(np.mean(quantum_w_scores))


# 1. SAVE CSV DaTA


# Data frame for accumulated scores
df_accumulated = pd.DataFrame({
    "Round": list(range(num_rounds + 1)),
    f"Classical_{num_classical_2_players}_Accumulated_Score": accum_classical_2,
    f"Classical_{num_variable_classical_players}_Accumulated_Score": accum_classical_variable,
    "Bell_Accumulated_Score_PureMeasurement": accum_quantum_bell,
    f"W_State_{num_w_players}Q_Accumulated_Score_PureMeasurement": accum_quantum_w,
})
csv_path_accumulated = os.path.join(OUT_DIR, "Accumulated_Scores_PureMeasurement.csv")
df_accumulated.to_csv(csv_path_accumulated, index=False)
print(f"Saved accumulated score data to: {csv_path_accumulated}")

# Data frame for effectiveness scores
df_effectiveness = pd.DataFrame({
    "Round": list(range(num_rounds)),
    f"Classical_{num_classical_2_players}_Effectiveness_Score": classical_2_scores,
    f"Classical_{num_variable_classical_players}_Effectiveness_Score": classical_variable_scores,
    "Bell_Effectiveness_Score_PureMeasurement": quantum_bell_scores,
    f"W_State_{num_w_players}Q_Effectiveness_Score_PureMeasurement": quantum_w_scores,
})
csv_path_effectiveness = os.path.join(OUT_DIR, "Effectiveness_Scores_PureMeasurement.csv")
df_effectiveness.to_csv(csv_path_effectiveness, index=False)
print(f"Saved effectiveness score data to: {csv_path_effectiveness}")

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
plt.plot(range(num_rounds + 1), accum_classical_2,
          label=f"Classical Team ({num_classical_2_players}C)", color='red', linestyle='--', linewidth=2)
plt.plot(range(num_rounds + 1), accum_classical_variable,
          label=f"Classical Team ({num_variable_classical_players}C)", color='red', linestyle='-', linewidth=2)
plt.plot(range(num_rounds + 1), accum_quantum_bell,
          label=f"Bell-State (2Q, Pure Measurement)", color='blue', linewidth=2)
plt.plot(range(num_rounds + 1), accum_quantum_w,
          label=f"W-State ({num_w_players}Q, Pure Measurement)", color='green', linewidth=2)
plt.xlabel("Round Number")
plt.ylabel("Accumulated Score")
plt.title("Accumulated Score Over Rounds (Ideal Conditions - Pure Measurement)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "RealWorldStrategy_Accumulated_Score.png"), dpi=300)

# --- Side-by-side histogram comparison ---
# Bin edges must cover the largest team size
max_score = max(num_classical_2_players, num_variable_classical_players, num_bell_players, num_w_players)
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
plt.hist(quantum_bell_scores, bins=bin_edges + 2*width, alpha=0.8, color='blue',
          label=f"Bell-State (2Q) (Mean: {mean_bell:.2f})",
          rwidth=width, align='left')
plt.hist(quantum_w_scores, bins=bin_edges + 3*width, alpha=0.8, color='green',
          label=f"W-State ({num_w_players}Q) (Mean: {mean_w:.2f})",
          rwidth=width, align='left')

plt.xlabel("Effectiveness Score")
plt.ylabel("Frequency")
plt.title("Effectiveness Comparison of Classical and Quantum Teams (Ideal Conditions - Pure Measurement)")
plt.xticks(np.arange(min_score, max_score + 1) + 1.5*width)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "RealWorldStrategy_Effectiveness_Comparison.png"), dpi=300)
