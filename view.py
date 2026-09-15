import numpy as np
from pysmps import smps_loader

filepath = "feasible/afiro.mps" 

mps_data = smps_loader.load_mps(filepath)

name = mps_data[0]  
c = mps_data[6]     
A = mps_data[7]     

rhs_name = mps_data[8][0]
b = mps_data[9][rhs_name]  

print(f"Model Name: {name}")
print(f"Constraint Matrix (A) shape: {A.shape}")
print(f"Objective Vector (c) shape: {c.shape}")
print(f"Bounds Vector (b) shape: {b.shape}")

print("\nFirst 5 values of objective vector c:")
print(c[:5])