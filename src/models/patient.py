"""
Patient data class
"""
from dataclasses import dataclass


@dataclass
class Patient:
    patient_id: str
    age: int
    gender: str
    cancer_type: str
    stage: str
    biomarkers: str  # Raw string from CSV, will be parsed
    location: str
    ecog_status: int
    prior_treatments: str  # Raw string from CSV, will be parsed

    def __post_init__(self):
        import pandas as pd
        
        # Parse biomarkers (semicolon-separated)
        if self.biomarkers and not pd.isna(self.biomarkers):
            self.biomarkers = [
                biomarker.strip() 
                for biomarker in self.biomarkers.split(';')
                if biomarker.strip()
            ]
        else:
            self.biomarkers = []
        
        # Parse prior_treatments (comma-separated)
        if self.prior_treatments and not pd.isna(self.prior_treatments):
            self.prior_treatments = [
                treatment.strip() 
                for treatment in self.prior_treatments.split(',')
                if treatment.strip()
            ]
        else:
            self.prior_treatments = []


    def to_dict(self) -> dict:
        """
        Convert patient data to a dictionary
        """
        return {
            "patient_id": self.patient_id,
            "age": self.age,
            "gender": self.gender,
            "cancer_type": self.cancer_type,
            "stage": self.stage,
            "biomarkers": self.biomarkers,
            "location": self.location,
            "ecog_status": self.ecog_status,
            "prior_treatments": self.prior_treatments
        }
