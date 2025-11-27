import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_stabilization_lag_distributions(df):
    plt.figure(figsize=(10,4))
    plt.hist(df["stabilization_lag"].dropna(), bins=30)
    plt.title("Stabilization Lag Distribution")
    plt.xlabel("Lag")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(10,4))
    plt.hist(df["stabil_ratio"].dropna(), bins=30)
    plt.title("Normalized Stabilization Ratio")
    plt.xlabel("Ratio")
    plt.ylabel("Count")
    plt.show()

def plot_variance_vs_lag(df):
    plt.figure(figsize=(7,6))
    plt.scatter(df["baseline_var"], df["stabilization_lag"], s=20)
    plt.title("Variance vs Stabilization Lag")
    plt.xlabel("Post-CP Variance")
    plt.ylabel("Stabilization Lag")
    plt.show()
