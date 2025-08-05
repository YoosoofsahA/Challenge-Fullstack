"""
Data loader for loading patient data
"""

import pandas as pd
from models.patient import Patient


def load_patient_data(patient_id: str) -> Patient:
    """
    Load patient data from the database
    """
    patient_data = pd.read_csv('Challenge-Fullstack/patients.csv')
    patient_row = patient_data[
        patient_data['patient_id'] == patient_id
    ].iloc[0]
    return Patient(**patient_row.to_dict())
