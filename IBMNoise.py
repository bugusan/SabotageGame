import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel

# === Reproducibility and Style Parameters ===
NP_RANDOM_SEED = 20251021
SHOTS = 1024 
SIMULATOR_SEED = 20251021
TRANSPILE_SEED = 20251021
np.random.seed(NP_RANDOM_SEED)

FONT_SCALE_FACTOR = 1.25 # Increase fonts by 25%

# Ensure output directory exists
OUT_DIR = "IBMNoise_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# === Game Parameters (Dynamic Player Counts) ===
num_rounds = 100
num_bell_players = 2 
num_classical_2_players = 2 
num_w_players = 3  # W-State 
num_variable_classical_players = 3 # Main classical team size (e.g., 5C)

# === Core Utility Functions (MOVED TO TOP TO PREVENT 'NOT DEFINED' ERROR) ===
def generate_army_defense():
    paths = ['A', 'B']
    defended_path = np.random.choice(paths)
    defense = {path: "Strong" if path == defended_path else "None" for path in paths}
    return defense

def classical_team_strategy(num_players): # Made dynamic
    return np.random.choice(['A', 'B'], num_players)

def choices_from_bitstring(bitstr, num_players):
    """Maps a single measured bitstring to sabotage choices."""
    bits = bitstr[::-1]
    return ['A' if b == '1' else 'B' for b in bits[:num_players]]

def evaluate_effectiveness(choices, army_defense):
    return sum(1 if army_defense[choice] == "None" else -1 for choice in choices)


# === Setup IBM Noise Model ===
try:
    from qiskit_ibm_runtime.fake_provider import FakeKyiv
except ImportError:
    from qiskit.providers.fake_provider import FakeKyiv

backend = FakeKyiv()
fake_noise_model = NoiseModel.from_backend(backend)
print(f"Using offline noise model from {backend.name}")

simulator = Aer.get_backend('qasm_simulator')


# === Quantum Execution Helper ===
def run_quantum_circuit_and_sample(qc, num_players, noise_model=None):
    """Runs circuit, samples one bitstring from the memory (1024 shots)."""
    tqc = transpile(qc, simulator, optimization_level=1, seed_transpiler=TRANSPILE_SEED)
    result = simulator.run(tqc, shots=SHOTS, noise_model=noise_model, seed_simulator=SIMULATOR_SEED, memory=True).result()
    sampled_bitstring = np.random.choice(result.get_memory())
    return choices_from_bitstring(sampled_bitstring, num_players)


# === Quantum Circuit for Bell State ===
def bell_state_circuit(noise_model=None):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return run_quantum_circuit_and_sample(qc, num_bell_players, noise_model)

# === Quantum Circuit for W-State (RY-CX Cascade - Generalized) ===
def w_state_circuit(num_qubits, noise_model=None):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.x(0) # Start from |100...0>
    
    # W-state preparation using RY-CX cascade (Generalized)
    for i in range(num_qubits - 1):
        theta = 2 * np.arccos(np.sqrt((num_qubits - i - 1) / (num_qubits - i)))
        qc.ry(theta, i)
        qc.cx(i, i + 1)
        
    qc.measure(range(num_qubits), range(num_qubits))
    return run_quantum_circuit_and_sample(qc, num_qubits, noise_model)


# === Run Simulation Function ===
def run_simulation(noise_model=None):
    # Initialize lists for 4 teams
    c2_scores, c_var_scores, quantum_bell_scores, quantum_w_scores = [], [], [], []
    accum_c2, accum_c_var, accum_bell, accum_w = [0], [0], [0], [0]
    
    np.random.seed(NP_RANDOM_SEED)

    for round_num in range(num_rounds):
        army_defense = generate_army_defense()

        # Classical (2 Players, 2C)
        c2_choice = classical_team_strategy(num_classical_2_players)
        c2_eff = evaluate_effectiveness(c2_choice, army_defense)
        c2_scores.append(c2_eff)
        accum_c2.append(accum_c2[-1] + c2_eff)

        # Classical (Variable Players, e.g., 5C)
        c_var_choice = classical_team_strategy(num_variable_classical_players)
        c_var_eff = evaluate_effectiveness(c_var_choice, army_defense)
        c_var_scores.append(c_var_eff)
        accum_c_var.append(accum_c_var[-1] + c_var_eff)

        # Quantum Bell
        quantum_bell_choice = bell_state_circuit(noise_model)
        quantum_bell_effectiveness = evaluate_effectiveness(quantum_bell_choice, army_defense)
        quantum_bell_scores.append(quantum_bell_effectiveness)
        accum_bell.append(accum_bell[-1] + quantum_bell_effectiveness)

        # Quantum W-State
        quantum_w_choice = w_state_circuit(num_w_players, noise_model)
        quantum_w_effectiveness = evaluate_effectiveness(quantum_w_choice, army_defense)
        quantum_w_scores.append(quantum_w_effectiveness)
        accum_w.append(accum_w[-1] + quantum_w_effectiveness)

    return c2_scores, c_var_scores, quantum_bell_scores, quantum_w_scores, accum_c2, accum_c_var, accum_bell, accum_w

# === Run Simulations with Fake Noise ===
c2_scores, c_var_scores, bell_scores, w_scores, acc_c2, acc_c_var, acc_bell, acc_w = run_simulation(fake_noise_model)


# === Calculate Mean Scores ===
mean_c2 = np.mean(c2_scores)
mean_c_var = np.mean(c_var_scores)
mean_bell = np.mean(bell_scores)
mean_w = np.mean(w_scores)


# ==============================================================================
# 1. SAVE CSV DATA
# ==============================================================================

# Data frame for accumulated scores
df_accumulated = pd.DataFrame({
    "Round": list(range(num_rounds + 1)),
    f"Classical_{num_classical_2_players}_Accumulated_Score": acc_c2,
    f"Classical_{num_variable_classical_players}_Accumulated_Score": acc_c_var,
    "Bell_Accumulated_Score_KyivNoise": acc_bell,
    f"W_State_{num_w_players}Q_Accumulated_Score_KyivNoise": acc_w,
})
csv_path_accumulated = os.path.join(OUT_DIR, "Accumulated_Scores_FakeKyiv.csv")
df_accumulated.to_csv(csv_path_accumulated, index=False)
print(f"Saved accumulated score data to: {csv_path_accumulated}")

# Data frame for effectiveness scores
df_effectiveness = pd.DataFrame({
    "Round": list(range(num_rounds)),
    f"Classical_{num_classical_2_players}_Effectiveness_Score": c2_scores,
    f"Classical_{num_variable_classical_players}_Effectiveness_Score": c_var_scores,
    "Bell_Effectiveness_Score_KyivNoise": bell_scores,
    f"W_State_{num_w_players}Q_Effectiveness_Score_KyivNoise": w_scores,
})
csv_path_effectiveness = os.path.join(OUT_DIR, "Effectiveness_Scores_FakeKyiv.csv")
df_effectiveness.to_csv(csv_path_effectiveness, index=False)
print(f"Saved effectiveness score data to: {csv_path_effectiveness}")


# ==============================================================================
# 2. PLOTTING WITH INCREASED FONT SIZE (25%)
# ==============================================================================

# Set default font sizes based on the scale factor
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

# === Plot Accumulated Scores ===
plt.figure(figsize=(10, 6))
plt.plot(acc_c2, label=f"Classical Team ({num_classical_2_players}C)", color='darkred', linestyle='--', linewidth=1.5)
plt.plot(acc_c_var, label=f"Classical Team ({num_variable_classical_players}C)", color='red', linestyle='-', linewidth=2)
plt.plot(acc_bell, label="Bell State (2Q, Kyiv Noise)", color='blue', linewidth=2)
plt.plot(acc_w, label=f"W-State ({num_w_players}Q, Kyiv Noise)", color='green', linewidth=2)

plt.xlabel("Round")
plt.ylabel("Accumulated Score")
plt.title("Accumulated Score Over Rounds with Kyiv Noise (Pure Measurement)")
plt.legend(loc="best") 
plt.tight_layout() 
plt.savefig(os.path.join(OUT_DIR, "IBMNoise_Accumulated_Score.png"), dpi=300)
# plt.show()

# === Plot Effectiveness Distribution ===
max_score = max(num_classical_2_players, num_variable_classical_players, num_bell_players, num_w_players)
min_score = -max_score
bin_edges = np.arange(min_score - 1, max_score + 2)
width = 0.20 # Adjust width for 4 groups

plt.figure(figsize=(10, 6))
plt.hist(c2_scores, bins=bin_edges + 0*width, alpha=0.7, color='red', hatch='//',
          label=f"Classical ({num_classical_2_players}C) (Mean: {mean_c2:.2f})",
          rwidth=width, align='left')
plt.hist(c_var_scores, bins=bin_edges + 1*width, alpha=0.8, color='red',
          label=f"Classical ({num_variable_classical_players}C) (Mean: {mean_c_var:.2f})",
          rwidth=width, align='left')
plt.hist(bell_scores, bins=bin_edges + 2*width, alpha=0.8, color='blue',
          label=f"Bell-State (2Q) (Mean: {mean_bell:.2f})",
          rwidth=width, align='left')
plt.hist(w_scores, bins=bin_edges + 3*width, alpha=0.8, color='green',
          label=f"W-State ({num_w_players}Q) (Mean: {mean_w:.2f})",
          rwidth=width, align='left')

plt.xlabel("Effectiveness Score")
plt.ylabel("Frequency")
plt.title("Effectiveness Comparison with Kyiv Noise (Pure Measurement)")
plt.xticks(np.arange(min_score, max_score + 1) + 1.5*width)
plt.legend(loc="best") 
plt.tight_layout() 
plt.savefig(os.path.join(OUT_DIR, "IBMNoise_Effectiveness_Comparison.png"), dpi=300)
# plt.show()