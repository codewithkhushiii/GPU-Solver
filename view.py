import numpy as np
from pysmps import smps_loader

filepath = "feasible/afiro.mps" 

# Capture the full tuple
mps_data = smps_loader.load_mps(filepath)

# Extract exactly what we need using the correct indices
name = mps_data[0]  # Index 0: Model name ('AFIRO')
c = mps_data[6]     # Index 6: Objective vector (already an ndarray)
A = mps_data[7]     # Index 7: Constraint matrix (already an ndarray)

# The Right-Hand Side (RHS / bounds) is stored in a dictionary at Index 9.
# The name of the RHS (e.g., 'B') is stored at Index 8.
rhs_name = mps_data[8][0]
b = mps_data[9][rhs_name]  # Fetch the specific array from the dictionary

# Verify everything works!
print(f"Model Name: {name}")
print(f"Constraint Matrix (A) shape: {A.shape}")
print(f"Objective Vector (c) shape: {c.shape}")
print(f"Bounds Vector (b) shape: {b.shape}")

print("\nFirst 5 values of objective vector c:")
print(c[:5])