"""
Main file for matching patients to clinical trials
"""

from data.data_loader import load_patient_data

# Load data
patient_data = load_patient_data('P005')

# Load patient data for specific patient
print(patient_data)

# Get trial data from biomcp

# Load trial data

# Match patients to trials using LLM to score and rank trials

# Evaluate matches using LLM

# Return results