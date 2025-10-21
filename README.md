
# Team-Based Quantum Sabotage Game
Reproducible Qiskit simulations for the paper "Team-Based Quantum Sabotage Game."
This repository contains the complete Python source code used to generate all figures and results for the manuscript: **"Team-Based Quantum Sabotage Game: Strategic Advantages of Entanglement and Superposition."**

The simulations use Qiskit to compare the effectiveness of classical teams (2C, 3C) against quantum-entangled teams (2Q Bell-state, 3Q W-state) in an adversarial sabotage scenario. We provide scripts for four distinct simulation types: a theoretical Hybrid Adaptive Heuristic (HAH), an ideal (noise-free) circuit simulation, simulations with standard noise models (SNM), and simulations with a calibrated noise model from real IBM Quantum hardware (`FakeKyiv`).

**Author:** Sinan Bugu


## 🚀 How to Run & Reproduce Results

All scripts are configured with fixed random seeds (e.g., `NP_RANDOM_SEED = 20251021`) to ensure 100% reproducibility.

### 1. Setup

Clone the repository and install the required packages.

```bash
# 1. Clone this repository
#    (Replace [YourUsername] with your GitHub username)
git clone [https://github.com/](https://github.com/)[YourUsername]/SabotageGame.git
cd SabotageGame

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# 3. Install all required packages
pip install -r requirements.txt

2. Run Simulations
You can run each script directly from your terminal. Each script will automatically create its own output directory (e.g., ClassicalStrategy_figures/) containing the .png plot files and .csv data files.

Bash

# Run the HAH Benchmark simulation (generates Figs 1-2)
python ClassicalStrategy.py

# Run the Ideal (Noise-Free) Circuit simulation (generates Figs 3-4)
python RealWorldStrategy.py

# Run the Standard Noise Model simulation (generates Figs 5-6)
python RealWordManualNoise.py

# Run the Hardware (Kyiv Noise) simulation (generates Figs 7-8)
python IBMNoise.py
After running all four scripts, you will have all the figures and raw data used in the paper.

📁 Repository Structure
ClassicalStrategy.py: Generates HAH benchmark results (Figs 1-2).

RealWorldStrategy.py: Generates ideal (noise-free) Qiskit circuit results (Figs 3-4).

RealWordManualNoise.py: Generates standard noise model (SNM) results (Figs 5-6).

IBMNoise.py: Generates calibrated hardware noise (FakeKyiv) results (Figs 7-8).

requirements.txt: A list of all necessary Python packages.

LICENSE: The MIT License for this source code.

circuits/: Contains circuit diagrams for the Bell and W3 states.

paper/: Contains the final PDF of the manuscript.

📈 Key Quantum Circuits
The quantum strategies rely on 2-qubit Bell states and 3-qubit W-states, the circuits attached in Circuits folder
