# Utility functions

# Imports
import numpy as np
import matplotlib.pyplot as plt


# Probability density function
def pdf(x, n_bins = 100):
    
    # Prey and predator densities
    u = x[:, 0]
    v = x[:, 1]
    
    # Predator and prey counts and bins
    u_counts, u_bins = np.histogram(u, bins=n_bins)
    v_counts, v_bins = np.histogram(v, bins=n_bins)

    # Normalization
    u_pdf = u_counts / (np.sum(u_counts) * (u_bins[1] - u_bins[0]))
    v_pdf = v_counts / (np.sum(v_counts) * (v_bins[1] - v_bins[0]))
    
    return u_pdf, u_bins, v_pdf, v_bins

    
# Simulation plotting function
def plot_simulation(x, t, x_det=None, t_det=None):
    
    # Compute pdf 
    u_pdf, u_bins, v_pdf, v_bins = pdf(x)
    
    # Create plot figure and axes
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    # Prey temporal evolution
    ax[0,0].plot(t, x[:, 0], label='Prey Density', color='red', zorder=3)
    if x_det is not None and t_det is not None:
        ax[0,0].plot(t_det, x_det[:, 0], label='Deterministic', color='black', linestyle='--', zorder=2)
    ax[0,0].set_title('Prey Density Temporal Evolution')
    ax[0,0].set_xlabel('Time (t)')
    ax[0,0].set_ylabel('Prey Density (u)')
    ax[0,0].set_xlim(0, None)
    ax[0,0].set_ylim(0, None)
    ax[0,0].grid(linestyle='--', zorder=1)
    ax[0,0].legend()

    # Predator temporal evolution
    ax[0,1].plot(t, x[:, 1], label='Predator Density', color='blue', zorder=3)
    if x_det is not None and t_det is not None:
        ax[0,1].plot(t_det, x_det[:, 1], label='Deterministic', color='black', linestyle='--', zorder=2)
    ax[0,1].set_title('Predator Density Temporal Evolution')
    ax[0,1].set_xlabel('Time (t)')
    ax[0,1].set_ylabel('Predator Density (v)')
    ax[0,1].set_xlim(0, None)
    ax[0,1].set_ylim(0, None)
    ax[0,1].grid(linestyle='--', zorder=1)
    ax[0,1].legend()

    # Prey PDF
    bin_centers = 0.5 * (u_bins[1:] + u_bins[:-1])
    ax[1,0].bar(bin_centers, u_pdf, width=(u_bins[1] - u_bins[0]), alpha=0.7, color='red', edgecolor='black', zorder=2)
    ax[1,0].set_title('Prey Density Distribution')
    ax[1,0].set_xlabel('Prey Density (u)')
    ax[1,0].set_ylabel('Probability Density')
    ax[1,0].grid(axis='y', linestyle='--', zorder=1)

    # Predator density PDF
    bin_centers = 0.5 * (v_bins[1:] + v_bins[:-1])
    ax[1,1].bar(bin_centers, v_pdf, width=(v_bins[1] - v_bins[0]), alpha=0.7, color='blue', edgecolor='black', zorder=2)
    ax[1,1].set_title('Predator Density Distribution')
    ax[1,1].set_xlabel('Predator Density (v)')
    ax[1,1].set_ylabel('Probability Density')
    ax[1,1].grid(axis='y', linestyle='--', zorder=1)

    plt.tight_layout()
    plt.show()

# White shot noise simulation plotting function
def plot_white_shot(x, t, impulses, x_det=None, t_det=None):
    
    # Compute pdf 
    u_pdf, u_bins, v_pdf, v_bins = pdf(x)
    
    # Create plot figure and axes
    fig, ax = plt.subplots(3, 2, figsize=(12, 9))

    # Prey temporal evolution
    ax[0,0].plot(t, x[:, 0], label='Prey Density', color='red', zorder=3)
    if x_det is not None and t_det is not None:
        ax[0,0].plot(t_det, x_det[:, 0], label='Deterministic', color='black', linestyle='--', zorder=2)
    ax[0,0].set_title('Prey Density Temporal Evolution')
    ax[0,0].set_xlabel('Time (t)')
    ax[0,0].set_ylabel('Prey Density (u)')
    ax[0,0].set_xlim(0, None)
    ax[0,0].set_ylim(0, None)
    ax[0,0].grid(linestyle='--', zorder=1)
    ax[0,0].legend()

    # Predator temporal evolution
    ax[0,1].plot(t, x[:, 1], label='Predator Density', color='blue', zorder=3)
    if x_det is not None and t_det is not None:
        ax[0,1].plot(t_det, x_det[:, 1], label='Deterministic', color='black', linestyle='--', zorder=2)
    ax[0,1].set_title('Predator Density Temporal Evolution')
    ax[0,1].set_xlabel('Time (t)')
    ax[0,1].set_ylabel('Predator Density (v)')
    ax[0,1].set_xlim(0, None)
    ax[0,1].set_ylim(0, None)
    ax[0,1].grid(linestyle='--', zorder=1)
    ax[0,1].legend()

    # Prey impulses
    ax[1,0].vlines(t, 0, impulses[:, 0], label='Prey Impulses', color='red', zorder=3)
    ax[1,0].set_title('Prey Impulses Over Time')
    ax[1,0].set_xlabel('Time (t)')
    ax[1,0].set_ylabel('Impulse Magnitude')
    ax[1,0].set_xlim(0, None)
    ax[1,0].grid(linestyle='--', zorder=1)
    
    # Predator impulses
    ax[1,1].vlines(t, 0, impulses[:, 1], label='Predator Impulses', color='blue', zorder=3)
    ax[1,1].set_title('Predator Impulses Over Time')
    ax[1,1].set_xlabel('Time (t)')
    ax[1,1].set_ylabel('Impulse Magnitude')
    ax[1,1].set_xlim(0, None)
    ax[1,1].grid(linestyle='--', zorder=1)

    # Prey PDF
    bin_centers = 0.5 * (u_bins[1:] + u_bins[:-1])
    ax[2,0].bar(bin_centers, u_pdf, width=(u_bins[1] - u_bins[0]), alpha=0.7, color='red', edgecolor='black', zorder=2)
    ax[2,0].set_title('Prey Density Distribution')
    ax[2,0].set_xlabel('Prey Density (u)')
    ax[2,0].set_ylabel('Probability Density')
    ax[2,0].grid(axis='y', linestyle='--', zorder=1)

    # Plot predator density PDF
    bin_centers = 0.5 * (v_bins[1:] + v_bins[:-1])
    ax[2,1].bar(bin_centers, v_pdf, width=(v_bins[1] - v_bins[0]), alpha=0.7, color='blue', edgecolor='black', zorder=2)
    ax[2,1].set_title('Predator Density Distribution')
    ax[2,1].set_xlabel('Predator Density (v)')
    ax[2,1].set_ylabel('Probability Density')
    ax[2,1].grid(axis='y', linestyle='--', zorder=1)

    plt.tight_layout()
    plt.show()