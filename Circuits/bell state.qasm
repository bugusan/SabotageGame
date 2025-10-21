OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

// |00> --H--> (|0>+|1>)/√2 on q0, then CNOT entangles
h q[0];
cx q[0], q[1];

barrier q;
measure q -> c;
