Problem Formulation: Compiling "Flag-by-Construction" ZX Spider Networks
=======================================

### **The Bipartite ZX Network (The Matrix Mapping)**

We are implementing a quantum code defined by a parity-check matrix (e.g., $H_x$) using a ZX-calculus spider network. The matrix defines a strict bipartite graph:
Z-Spiders (Rows): Each row in the matrix represents a distinct Z-spider (implemented as a specific cat state).
X-Spiders (Columns): Any column that sums to $n>1$ represents an X-spider (also implemented as a cat state, with $n+1$ legs). Columns summing to 1 are simply unshared output legs for their respective Z-spider.
Wires (The 1s): A 1 at row i and column j indicates a wire connecting Z-spider i directly to X-spider j. There are no Z-Z or X-X connections.

### The Local Constraints (Cat State Implementations)
Every spider (whether Z or X) corresponds to a cat state.
We have known, fixed circuit implementations for these cat states.
These implementations enforce a strict causal partial ordering on their internal CNOT gates to function correctly.
We can represent each cat state's internal timing as a weighted Directed Acyclic Graph (DAG), where nodes are CNOTs and weights represent temporal delays (e.g., time taken by hidden flag qubits).

### Fusing the Wires (The Core Complexity)
This is where the local implementations meet the global structure.
A "wire" connecting a Z-spider and an X-spider is physically realized as a single CNOT gate.
This means a specific "leg" CNOT in the Z-spider's internal DAG and a specific "leg" CNOT in the X-spider's internal DAG are actually the exact same physical gate.
To build the global circuit, we must merge these corresponding CNOT nodes. Fusing these nodes glues the many local DAGs together into one massive global dependency graph.

### The Optimization Problem (Causality and Minimal Depth)
When we fuse these DAGs together via the shared CNOTs, we are combining many different partial orderings.
The Causality Constraint: The final, fused global graph must remain a strict Directed Acyclic Graph (DAG). If merging the nodes creates a cycle (a closed time-like loop), the implementation is physically impossible. We need to find an appropriate way to order these connections so the whole system is causal.

The Objective: Assuming we can form a valid DAG, we want to find the configuration that results in the minimal circuit depth (i.e., minimizing the longest weighted path through the global DAG).