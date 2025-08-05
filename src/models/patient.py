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
        # Parse biomarkers (semicolon-separated)
        if self.biomarkers:
            self.biomarkers = [
                biomarker.strip() 
                for biomarker in self.biomarkers.split(';')
            ]
        else:
            self.biomarkers = []
        
        # Parse prior_treatments (comma-separated)
        if self.prior_treatments:
            self.prior_treatments = [
                treatment.strip() 
                for treatment in self.prior_treatments.split(',')
            ]
        else:
            self.prior_treatments = []
