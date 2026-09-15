import numpy as np
import torch
from pysmps import smps_loader
import time
import os
import csv


def preprocess_mps(filepath):
    print(f"Parsing MPS file: {filepath}...")
    mps_data = smps_loader.load_mps(filepath)

    name = mps_data[0]
    row_types = mps_data[5] 
    c_raw = mps_data[6]
    A_raw = mps_data[7]
    rhs_name = mps_data[8][0]
    b_raw = mps_data[9][rhs_name]

    m, n = A_raw.shape

    A_cols = list(A_raw.T)
    c_new = list(c_raw)

    added_slacks = 0
    for i, r_type in enumerate(row_types):
        r_type_str = r_type.decode('utf-8') if isinstance(r_type, bytes) else r_type

        if r_type_str == 'L':
            col = np.zeros(m)
            col[i] = 1.0
            A_cols.append(col)
            c_new.append(0.0)
            added_slacks += 1

        elif r_type_str == 'G':
            col = np.zeros(m)
            col[i] = -1.0
            A_cols.append(col)
            c_new.append(0.0)
            added_slacks += 1

    A_std = np.column_stack(A_cols)
    c_std = np.array(c_new)
    b_std = b_raw

    print(f"Added {added_slacks} slack/surplus variables.")
    print(f"Original A shape: {A_raw.shape} | Standardized A shape: {A_std.shape}\n")
    return c_std, A_std, b_std


def pytorch_ipm_solver(c_np, A_np, b_np, max_iter=100, tol=1e-8, sigma=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Engine on: {device} ---")

    c = torch.tensor(c_np, dtype=torch.float64, device=device).unsqueeze(1)
    A = torch.tensor(A_np, dtype=torch.float64, device=device)
    b = torch.tensor(b_np, dtype=torch.float64, device=device).unsqueeze(1)

    m, n = A.shape

    x = torch.ones((n, 1), dtype=torch.float64, device=device)
    y = torch.zeros((m, 1), dtype=torch.float64, device=device)
    s = torch.ones((n, 1), dtype=torch.float64, device=device)

    b_norm = max(1.0, torch.linalg.norm(b).item())
    c_norm = max(1.0, torch.linalg.norm(c).item())

    for iteration in range(max_iter):
        mu = (torch.dot(x.view(-1), s.view(-1)) / n).item()

        r_b = A @ x - b
        r_c = A.T @ y + s - c

        r_b_rel = (torch.linalg.norm(r_b) / b_norm).item()
        r_c_rel = (torch.linalg.norm(r_c) / c_norm).item()

        if iteration % 5 == 0 or mu < tol:
            current_obj = (c.T @ x).item()
            print(f"Iter {iteration:<3} | Objective = {current_obj:<14.6f} | "
                  f"Gap = {mu:.3e} | |r_b| = {r_b_rel:.3e} | |r_c| = {r_c_rel:.3e}")

        if mu < tol and r_b_rel < tol and r_c_rel < tol:
            print(f"Converged successfully at iteration {iteration}!")
            break

        S_inv_X = x / s  # D = X S^-1

        reg_val = max(1e-10, mu * 1e-6)
        reg_matrix = reg_val * torch.eye(m, dtype=torch.float64, device=device)
        M = A @ torch.diag(S_inv_X.view(-1)) @ A.T + reg_matrix

        rhs = -r_b - A @ (S_inv_X * r_c) + A @ (x - (sigma * mu) / s)

        try:
            delta_y = torch.linalg.solve(M, rhs)
        except RuntimeError as e:
            print(f"Numerical instability at iter {iteration}: {e}")
            break

        delta_s = -r_c - A.T @ delta_y
        delta_x = -x + (sigma * mu) / s - (x / s) * delta_s


        alpha_primal = 1.0
        alpha_dual = 1.0
        while torch.any((x + alpha_primal * delta_x) <= 0) and alpha_primal > 1e-12:
            alpha_primal *= 0.9
        while torch.any((s + alpha_dual * delta_s) <= 0) and alpha_dual > 1e-12:
            alpha_dual *= 0.9

        x = x + 0.99 * alpha_primal * delta_x
        y = y + 0.99 * alpha_dual * delta_y
        s = s + 0.99 * alpha_dual * delta_s

    final_obj = (c.T @ x).item()
    return final_obj

if __name__ == "__main__":
    folder_path = "feasible"
    output_file = "benchmark_results.csv"
    
    mps_files = [f for f in os.listdir(folder_path) if f.endswith('.mps')]
    

    with open(output_file, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
  
        csv_writer.writerow(["Filename", "Status", "Objective Value", "Time (s)"])
        
        print(f"{'Filename':<15} | {'Status':<15} | {'Objective Value':<15} | {'Time (s)'}")
        print("-" * 65)
        
        for filename in mps_files:
            filepath = os.path.join(folder_path, filename)
            
       
            file_size_kb = os.path.getsize(filepath) / 1024
            if file_size_kb > 500:  
                print(f"{filename:<15} | {'Skipped (Large)':<15} | {'N/A':<15} | N/A")
                csv_writer.writerow([filename, "Skipped (Large)", "N/A", "N/A"])
                continue
                
            try:
                c_std, A_std, b_std = preprocess_mps(filepath)

                start_time = time.perf_counter()
                
                final_value = pytorch_ipm_solver(c_std, A_std, b_std)
                
                end_time = time.perf_counter()
                compute_time = end_time - start_time
                
                print(f"{filename:<15} | {'Success':<15} | {final_value:<15.4f} | {compute_time:.4f}")
                csv_writer.writerow([filename, "Success", f"{final_value:.4f}", f"{compute_time:.4f}"])
                
            except Exception as e:
                print(f"{filename:<15} | {'Failed':<15} | {'N/A':<15} | N/A")
                csv_writer.writerow([filename, "Failed", "N/A", "N/A"])
                
    print(f"\n Benchmarking complete! Results successfully saved to '{output_file}'.")