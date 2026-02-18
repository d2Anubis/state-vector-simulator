
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_bloch_multivector(state: np.ndarray, title: str = "Bloch Sphere", filename: str = None):
    """
    Plots the Bloch vector for a single-qubit state.
    Args:
        state: A numpy array of shape (2,) representing alpha|0> + beta|1>.
    """
    if state.shape != (2,):
        raise ValueError("Bloch sphere visualization currently only supports single-qubit states (size 2).")

    # Normalize if not already
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0):
        state = state / norm

    alpha = state[0]
    beta = state[1]

    # Calculate Bloch vector components
    # x = 2 * Re(alpha* . beta)
    # y = 2 * Im(alpha* . beta)
    # z = |alpha|^2 - |beta|^2
    
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha)**2 - np.abs(beta)**2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # Draw Sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = np.cos(u)*np.sin(v)
    ys = np.sin(u)*np.sin(v)
    zs = np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.4)

    # Draw Axes
    ax.plot([-1, 1], [0, 0], [0, 0], 'k-', lw=1, alpha=0.5) # X
    ax.plot([0, 0], [-1, 1], [0, 0], 'k-', lw=1, alpha=0.5) # Y
    ax.plot([0, 0], [0, 0], [-1, 1], 'k-', lw=1, alpha=0.5) # Z
    
    # Labels
    ax.text(1.1, 0, 0, "x (|0>+|1>)", fontsize=10)
    ax.text(0, 1.1, 0, "y", fontsize=10)
    ax.text(0, 0, 1.1, "|0>", fontsize=10)
    ax.text(0, 0, -1.1, "|1>", fontsize=10)

    # Draw Vector
    ax.quiver(0, 0, 0, x, y, z, color='r', length=1.0, normalize=False, lw=2)
    
    # Point
    ax.scatter([x], [y], [z], color='r', s=50)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Ensure aspect ratio is equal (simulated)
    ax.set_box_aspect([1,1,1])
    
    if filename:
        plt.savefig(filename)
        print(f"Saved Bloch sphere to {filename}")
        plt.close(fig)
        return None
    else:
        return fig
