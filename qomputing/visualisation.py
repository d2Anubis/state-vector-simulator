"""
Visualization tools for quantum circuits.
"""
from typing import Dict, List, Tuple, Optional, Any, Set
from qomputing.circuit import QuantumCircuit, Gate

def draw_circuit(circuit: QuantumCircuit) -> str:
    """
    Draws a text representation of the quantum circuit using a compact Qiskit-like style.
    """
    if circuit.num_qubits == 0:
        return "Empty Circuit"

    # Initialize grid: 3 lines per qubit
    grid = []
    for i in range(circuit.num_qubits):
        label = f"q{i}: "
        grid.append([" " * len(label)])   # Top
        grid.append([label])              # Mid
        grid.append([" " * len(label)])   # Bot

    # Classical register line buffer
    # We'll build it progressively or at the end.
    # Let's track a separate 'c_line' string builder
    c_label = f"c: {circuit.num_clbits}/ " if circuit.num_clbits > 0 else "c: "
    c_line_parts = [c_label] 
    
    def current_len(q): 
        return len("".join(grid[3 * q + 1]))
        
    def current_c_len():
        return len("".join(c_line_parts))

    def pad_all(target_len):
        # Pad qubits
        for q in range(circuit.num_qubits):
            curr = current_len(q)
            diff = target_len - curr
            if diff > 0:
                grid[3 * q].append(" " * diff)
                grid[3 * q + 1].append("─" * diff)
                grid[3 * q + 2].append(" " * diff)
        
        # Pad classical line
        curr_c = current_c_len()
        diff_c = target_len - curr_c
        if diff_c > 0:
            c_line_parts.append("═" * diff_c)

    # Helper to determine gate properties
    def get_gate_info(gate: Gate) -> Tuple[int, str]:
        label = gate.name.upper()
        # User explicitly requested to remove params (like 3.14) from labels
        # So we just use the name.
        
        width = len(label) + 2 # [ H ]
        if width < 5: width = 5 # Minimum width
        if width % 2 == 0: width += 1 # Odd width for centering
        return width, label

    # Process Gates
    for gate in circuit.gates:
        involved = list(gate.targets) + list(gate.controls)
        if not involved: continue
        
        min_q, max_q = min(involved), max(involved)
        span = range(min_q, max_q + 1)
        
        # Determine start position
        start_x = max(current_len(q) for q in span) + 1
        # Sync with classical line
        start_x = max(start_x, current_c_len())
        
        pad_all(start_x)
        
        width, label = get_gate_info(gate)
        mid_idx = width // 2
        
        for q in range(circuit.num_qubits):
            if q in span:
                top_seg = list(" " * width)
                mid_seg = list("─" * width)
                bot_seg = list(" " * width)
                
                if q in gate.controls:
                    mid_seg[mid_idx] = "■"
                    if q > min_q: top_seg[mid_idx] = "│"
                    if q < max_q: bot_seg[mid_idx] = "│"
                elif q in gate.targets:
                    top_seg = list("┌" + "─"*(width-2) + "┐")
                    bot_seg = list("└" + "─"*(width-2) + "┘")
                    mid_seg = list("┤" + label.center(width-2) + "├")
                    if q > min_q: top_seg[mid_idx] = "┴" # Connect from top
                    if q < max_q: bot_seg[mid_idx] = "┬" # Connect to bottom
                else:
                    # Vertical pass-through
                    top_seg[mid_idx] = "│"
                    mid_seg[mid_idx] = "┼"
                    bot_seg[mid_idx] = "│"
                
                grid[3 * q].append("".join(top_seg))
                grid[3 * q + 1].append("".join(mid_seg))
                grid[3 * q + 2].append("".join(bot_seg))
            else:
                # Pad others
                grid[3 * q].append(" " * width)
                grid[3 * q + 1].append("─" * width)
                grid[3 * q + 2].append(" " * width)
        
        # Pad classical line for this gate
        c_line_parts.append("═" * width)

    # Process Measurements
    if circuit._measurements:
        # Align all
        start_x = max(current_len(q) for q in range(circuit.num_qubits))
        start_x = max(start_x, current_c_len())
        pad_all(start_x)
        
        for q, c in circuit._measurements:
            width = 5
            
            # For each qubit column
            for idx in range(circuit.num_qubits):
                if idx == q:
                    grid[3*idx].append("┌───┐")
                    grid[3*idx+1].append("┤ M ├")
                    grid[3*idx+2].append("└─╥─┘")
                elif idx > q:
                    grid[3*idx].append("  ║  ")
                    grid[3*idx+1].append("──╫──")
                    grid[3*idx+2].append("  ║  ")
                else:
                    # Pad
                    grid[3*idx].append(" " * width)
                    grid[3*idx+1].append("─" * width)
                    grid[3*idx+2].append(" " * width)
            
            # Classical line connection
            # "══╩══"
            segment = list("═" * width)
            segment[2] = "╩"
            c_line_parts.append("".join(segment))

    # Finalize
    max_len = max(current_len(q) for q in range(circuit.num_qubits))
    pad_all(max_len) # Ensure c_line is fully extended if needed? 
    # Actually pad_all checks c_line length too.
    
    result = []
    # Top margin
    result.append(" " * max_len)
    
    for row in grid:
        result.append("".join(row))
    
    # Add classical line if used
    if circuit.num_clbits > 0 or circuit._measurements:
        result.append("".join(c_line_parts))

    return "\n".join(result)