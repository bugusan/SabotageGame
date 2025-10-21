OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// Start from |100>
x q[0];

// Distribute the single excitation across the chain
// theta0 = 2*arccos(1/sqrt(3)) ≈ 1.9106332362490184
ry(1.9106332362490184) q[0];
cx q[0], q[1];

// theta1 = 2*arccos(1/sqrt(2)) = pi/2 ≈ 1.5707963267948966
ry(1.5707963267948966) q[1];
cx q[1], q[2];

barrier q;
measure q -> c;

