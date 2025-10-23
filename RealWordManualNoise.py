import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, pauli_error

# === Reproducibility and Style Parameters ===
NP_RANDOM_SEED = 20251021
SHOTS = 1024 
SIMULATOR_SEED = 20251021
TRANSPILE_SEED = 20251021
np.random.seed(NP_RANDOM_SEED)

FONT_SCALE_FACTOR = 1.25 

# Ensure output directory exists
OUT_DIR = "RealWorldManualNoise_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# === Game Parameters (Dynamic Player Counts) ===
num_rounds = 100
num_bell_players = 2 
num_classical_2_players = 2 
num_w_players = 3  # W-Stte 
num_variable_classical_players = 3 

simulator = Aer.get_backend("aer_simulator")

# === Helper Functions for Reproducibility ===
def choices_from_bitstring(bitstr, num_players):
    """
    Maps a single measured bitstring to sabotage choices.
    Convention: '1' -> A, '0' -> B.
    Bitstring is reversed to map q[i] to Player i.
    """
    bits = bitstr[::-1]
    return ['A' if b == '1' else 'B' for b in bits[:num_players]]

# === Circuits (Pure Measurement - Ideal Conditions) ===

def generate_bell_state():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return transpile(qc, simulator, optimization_level=1, seed_transpiler=TRANSPILE_SEED)

def generate_w_state(num_qubits):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.x(0)  # Start from |10...0>
    
    # === GENERALIZED R-Y/CX CASCADE FOR W-STATE PREPARATION ===
    for i in range(num_qubits - 1):
        # Angle formula: theta_k = 2 * arccos(sqrt((N - k - 1) / (N - k)))
        theta = 2 * np.arccos(np.sqrt((num_qubits - i - 1) / (num_qubits - i)))
        
        qc.ry(theta, i)
        qc.cx(i, i + 1)
        
    qc.measure(range(num_qubits), range(num_qubits))
    return transpile(qc, simulator, optimization_level=1, seed_transpiler=TRANSPILE_SEED)

# === Quantum Strategy Execution (Deterministic Sampling) ===

def run_quantum_circuit_and_sample(circuit, num_players, noise_model=None):
    """Runs circuit, samples one bitstring from the memory (1024 shots)."""
    job = simulator.run(circuit, shots=SHOTS, seed_simulator=SIMULATOR_SEED, memory=True, noise_model=noise_model)
    result = job.result()
    
    memory = result.get_memory() 
    sampled_bitstring = np.random.choice(memory)

    return choices_from_bitstring(sampled_bitstring, num_players)

# === Strategies ===
def generate_army_defense():
    paths = ['A', 'B']
    defended = np.random.choice(paths)
    return {path: ("Strong" if path == defended else "None") for path in paths}

def classical_team_strategy(num_players):
    return np.random.choice(['A', 'B'], num_players)

def bell_state_strategy(noise_model=None):
    return run_quantum_circuit_and_sample(generate_bell_state(), num_bell_players, noise_model)

def w_state_strategy(noise_model=None):
    return run_quantum_circuit_and_sample(generate_w_state(num_w_players), num_w_players, noise_model)

def evaluate_effectiveness(choices, army_defense):
    return sum(1 if army_defense[c] == "None" else -1 for c in choices)

# === Define Noise Models (Standard Noise) ===
one_q_basis = ['id', 'sx', 'x', 'rz']
two_q_basis = ['cx']

# Depolarizing Noise (0.05)
depolarizing_noise = NoiseModel()
depolarizing_noise.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), one_q_basis)
depolarizing_noise.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), two_q_basis)

# Amplitude Damping Noise (0.05)
amplitude_damping_noise = NoiseModel()
amplitude_damping_noise.add_all_qubit_quantum_error(amplitude_damping_error(0.05), one_q_basis)

# Bit-Flip Noise (0.05)
bit_flip_noise = NoiseModel()
bit_flip_noise.add_all_qubit_quantum_error(pauli_error([('X', 0.05), ('I', 0.95)]), one_q_basis)


# === Simulation Loop ===
def run_full_simulation_set(noise_model=None):
    classical_2_scores, classical_variable_scores, quantum_w_scores, quantum_bell_scores = [], [], [], []
    accum_classical_2, accum_classical_variable, accum_quantum_w, accum_quantum_bell = [0], [0], [0], [0]
    
    np.random.seed(NP_RANDOM_SEED)

    for rnd in range(num_rounds):
        defense = generate_army_defense()

        # Classical (2 Players)
        c2_choice = classical_team_strategy(num_classical_2_players)
        c2_eff = evaluate_effectiveness(c2_choice, defense)
        classical_2_scores.append(c2_eff)
        accum_classical_2.append(accum_classical_2[-1] + c2_eff)

        # Classical (Variable Players)
        c_variable_choice = classical_team_strategy(num_variable_classical_players)
        c_variable_eff = evaluate_effectiveness(c_variable_choice, defense)
        classical_variable_scores.append(c_variable_eff)
        accum_classical_variable.append(accum_classical_variable[-1] + c_variable_eff)

        # Bell (2Q)
        bell = bell_state_strategy(noise_model)
        beff = evaluate_effectiveness(bell, defense)
        quantum_bell_scores.append(beff)
        accum_quantum_bell.append(accum_quantum_bell[-1] + beff)

        # W-state (N Qubits)
        w = w_state_strategy(noise_model)
        weff = evaluate_effectiveness(w, defense)
        quantum_w_scores.append(weff)
        accum_quantum_w.append(accum_quantum_w[-1] + weff)

    return (classical_2_scores, classical_variable_scores, quantum_w_scores, quantum_bell_scores,
            accum_classical_2, accum_classical_variable, accum_quantum_w, accum_quantum_bell)

# --- Run All Scenarios ---

# 1. No Noise (Ideal Baseline)
c2_scores, c_var_scores, w_scores, bell_scores, acc_c2, acc_c_var, acc_w, acc_bell = run_full_simulation_set(noise_model=None)

# 2. Depolarizing Noise
_, _, w_depol_scores, bell_depol_scores, _, _, acc_w_depol, acc_bell_depol = run_full_simulation_set(depolarizing_noise)

# 3. Amplitude Damping Noise
_, _, w_amp_scores, bell_amp_scores, _, _, acc_w_amp, acc_bell_amp = run_full_simulation_set(amplitude_damping_noise)

# 4. Bit-Flip Noise
_, _, w_bitflip_scores, bell_bitflip_scores, _, _, acc_w_bitflip, acc_bell_bitflip = run_full_simulation_set(bit_flip_noise)


# === Statistics (For Plotting) ===
mean_c2 = np.mean(c2_scores)
mean_c_var = np.mean(c_var_scores)

# ==============================================================================
# 1. SAVE CSV DATA
# ==============================================================================

df_accumulated = pd.DataFrame({
    "Round": list(range(num_rounds + 1)),
    f"Classical_{num_classical_2_players}_Accumulated": acc_c2,
    f"Classical_{num_variable_classical_players}_Accumulated": acc_c_var,
    "Bell_NoNoise_Accumulated": acc_bell,
    f"W_State_{num_w_players}Q_NoNoise_Accumulated": acc_w,
    "Bell_Depolarizing_Accumulated": acc_bell_depol,
    f"W_State_{num_w_players}Q_Depolarizing_Accumulated": acc_w_depol,
    "Bell_AmplitudeDamping_Accumulated": acc_bell_amp,
    f"W_State_{num_w_players}Q_AmplitudeDamping_Accumulated": acc_w_amp,
    "Bell_BitFlip_Accumulated": acc_bell_bitflip,
    f"W_State_{num_w_players}Q_BitFlip_Accumulated": acc_w_bitflip,
})
csv_path_accumulated = os.path.join(OUT_DIR, "Accumulated_Scores_ManualNoise.csv")
df_accumulated.to_csv(csv_path_accumulated, index=False)
print(f"Saved accumulated score data to: {csv_path_accumulated}")

# Save effectiveness scores for histogram statistics
df_effectiveness = pd.DataFrame({
    "C2_NoNoise": c2_scores,
    "C_Var_NoNoise": c_var_scores,
    "Bell_NoNoise": bell_scores,
    f"W{num_w_players}Q_NoNoise": w_scores,
    "Bell_Depolarizing": bell_depol_scores,
    f"W{num_w_players}Q_Depolarizing": w_depol_scores,
    "Bell_AmplitudeDamping": bell_amp_scores,
    f"W{num_w_players}Q_AmplitudeDamping": w_amp_scores,
    "Bell_BitFlip": bell_bitflip_scores,
    f"W{num_w_players}Q_BitFlip": w_bitflip_scores,
})
csv_path_effectiveness = os.path.join(OUT_DIR, "Effectiveness_Scores_ManualNoise_HistData.csv")
df_effectiveness.to_csv(csv_path_effectiveness, index=False)
print(f"Saved effectiveness histogram data to: {csv_path_effectiveness}")

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


# --- Accumulated Scores Plot ---
plt.figure(figsize=(10, 6))
plt.plot(acc_c2, label=f"Classical ({num_classical_2_players}C) (No Noise)", color='darkred', linestyle='--', linewidth=1.5)
plt.plot(acc_c_var, label=f"Classical ({num_variable_classical_players}C) (No Noise)", color='red', linestyle='-', linewidth=2)
plt.plot(acc_bell, label="Bell (Pure Measurement, No Noise)", color='blue', linewidth=2)
plt.plot(acc_w, label=f"W-State ({num_w_players}Q, Pure Measurement, No Noise)", color='green', linewidth=2)

plt.plot(acc_bell_depol, '--', label="Bell (Depolarizing)", color='cyan', linewidth=1)
plt.plot(acc_w_depol, '--', label="W-State (Depolarizing)", color='lime', linewidth=1)
plt.plot(acc_bell_amp, '--', label="Bell (Amplitude Damping)", color='purple', linewidth=1)
plt.plot(acc_w_amp, '--', label="W-State (Amplitude Damping)", color='olive', linewidth=1)
plt.plot(acc_bell_bitflip, '--', label="Bell (Bit-Flip)", color='magenta', linewidth=1)
plt.plot(acc_w_bitflip, '--', label="W-State (Bit-Flip)", color='orange', linewidth=1)

plt.xlabel("Round")
plt.ylabel("Accumulated Score")
plt.title("Accumulated Score Over Rounds with Different Noise Models (Pure Measurement)")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "RealWorldManualNoise_Accumulated_Score.png"), dpi=300)

# --- Histogram Comparison ---
def stats(arr):
    mu = np.mean(arr)
    sd = np.std(arr)
    pos = 100 * np.mean(np.array(arr) > 0)
    return mu, sd, pos

def pretty_label(name, arr):
    mu, sd, pos = stats(arr)
    return f"{name}\nμ={mu:.2f}, σ={sd:.2f}, P(+)= {pos:.0f}%"

# All relevant scores for the histogram
jls_extract_var = num_w_players
all_scores = [
    (c2_scores, f"Classical ({num_classical_2_players}C, No Noise)", 'red', '//', 0.7),
    (c_var_scores, f"Classical ({num_variable_classical_players}C, No Noise)", 'red', '-', 0.8),
    (bell_scores, "Bell (No Noise)", 'blue', None, 0.8),
    (w_scores, f"W-State ({num_w_players}Q, No Noise)", 'green', None, 0.8),
    (bell_depol_scores, "Bell (Depolarizing)", 'cyan', None, 0.6),
    (w_depol_scores, f"W-State ({num_w_players}Q, Depolarizing)", 'lime', None, 0.6),
    (bell_amp_scores, "Bell (Amplitude Damping)", 'purple', None, 0.6),
    (w_amp_scores, f"W-State ({num_w_players}Q, Amplitude Damping)", 'olive', None, 0.6),
    (bell_bitflip_scores, "Bell (Bit-Flip)", 'magenta', None, 0.6),
 (w_bitflip_scores, f"W-State ({jls_extract_var}Q, Bit-Flip)", 'orange', None, 0.6)
]

# Calculate max/min score for binning
all_data = np.concatenate([s[0] for s in all_scores if isinstance(s, tuple)]) # Handle the dictionary entry correctly
min_hist_score, max_hist_score = int(np.min(all_data)), int(np.max(all_data))
bin_edges = np.arange(min_hist_score - 1, max_hist_score + 2)
width = 0.08 # Adjusted for 10 sets of data

plt.figure(figsize=(14, 8))

# Map the scores list to a plotting list
plotting_data = [
    (c2_scores, f"Classical ({num_classical_2_players}C, No Noise)", 'red', '//', 0.7),
    (c_var_scores, f"Classical ({num_variable_classical_players}C, No Noise)", 'darkred', '-', 0.8),
    (bell_scores, "Bell (No Noise)", 'blue', None, 0.8),
    (w_scores, f"W-State ({num_w_players}Q, No Noise)", 'green', None, 0.8),
    (bell_depol_scores, "Bell (Depolarizing)", 'cyan', None, 0.6),
    (w_depol_scores, f"W-State ({num_w_players}Q, Depolarizing)", 'lime', None, 0.6),
    (bell_amp_scores, "Bell (Amplitude Damping)", 'purple', None, 0.6),
    (w_amp_scores, f"W-State ({num_w_players}Q, Amplitude Damping)", 'olive', None, 0.6),
    (bell_bitflip_scores, "Bell (Bit-Flip)", 'magenta', None, 0.6),
    (w_bitflip_scores, f"W-State ({num_w_players}Q, Bit-Flip)", 'orange', None, 0.6),
]


for idx, (scores, name, color, hatch, alpha) in enumerate(plotting_data):
    plt.hist(scores, bins=bin_edges + idx * width, alpha=alpha, color=color, hatch=hatch,
             label=pretty_label(name, scores), rwidth=width, align='left')

plt.xlabel("Effectiveness Score")
plt.ylabel("Frequency")
plt.title("Effectiveness Comparison: Noise-Free and Noisy Cases (Pure Measurement)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc="upper left") 
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "RealWorldManualNoise_Effectiveness_Comparison.png"), dpi=300)
