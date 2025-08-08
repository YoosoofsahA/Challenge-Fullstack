"""
Evaluation scripts for LLM matching of patients to clinical trials - Medical Validity
"""

import statistics
from typing import Any


class TrialMatchingEvaluator:
    """
    Evaluator for LLM-based patient to trial matching results
    """
    
    def __init__(self):
        # TODO: Get this information from medical documantation/database
        raw_biomarker_mappings = {
            'HER2+': ['trastuzumab', 'pertuzumab', 'ado-trastuzumab', 'fam-trastuzumab'],
            'EGFR+': ['erlotinib', 'gefitinib', 'osimertinib', 'afatinib'],
            'KRAS G12C': ['sotorasib', 'adagrasib'],
            'BRAF V600E': ['vemurafenib', 'dabrafenib', 'encorafenib'],
            'ALK+': ['crizotinib', 'alectinib', 'ceritinib', 'brigatinib'],
            'ROS1+': ['crizotinib', 'entrectinib'],
            'NTRK+': ['larotrectinib', 'entrectinib'],
            'PD-L1+': ['pembrolizumab', 'nivolumab', 'atezolizumab', 'durvalumab'],
            'MSI-H': ['pembrolizumab', 'nivolumab'],
            'BRCA1/2': ['olaparib', 'rucaparib', 'niraparib', 'talazoparib'],
            'PIK3CA': ['alpelisib'],
            'CDK4/6': ['palbociclib', 'ribociclib', 'abemaciclib']
        }
        
        # Normalize biomarker keys for consistent lookup
        normalized_biomarker_mappings = {}
        for biomarker, drugs in raw_biomarker_mappings.items():
            normalized_key = biomarker.strip().replace(' ', '').lower()
            normalized_biomarker_mappings[normalized_key] = drugs
        
        self.medical_knowledge_base = {
            'biomarker_drug_mappings': normalized_biomarker_mappings,
            'cancer_synonyms': {
                'breast cancer': ['breast carcinoma', 'mammary carcinoma', 'breast neoplasm'],
                'lung cancer': ['pulmonary carcinoma', 'lung carcinoma', 'bronchogenic carcinoma', 'nsclc', 'non-small cell lung cancer'],
                'colorectal cancer': ['colon cancer', 'rectal cancer', 'crc', 'colorectal carcinoma'],
                'melanoma': ['malignant melanoma', 'cutaneous melanoma'],
                'ovarian cancer': ['ovarian carcinoma', 'ovarian neoplasm'],
                'prostate cancer': ['prostate carcinoma', 'prostatic carcinoma'],
                'pancreatic cancer': ['pancreatic carcinoma', 'pancreatic adenocarcinoma'],
                'gastric cancer': ['stomach cancer', 'gastric carcinoma'],
                'liver cancer': ['hepatocellular carcinoma', 'hcc', 'hepatic carcinoma'],
                'kidney cancer': ['renal cell carcinoma', 'rcc', 'renal carcinoma'],
                'bladder cancer': ['urothelial carcinoma', 'bladder carcinoma'],
                'head and neck cancer': ['hnsc', 'squamous cell carcinoma of head and neck'],
                'glioblastoma': ['gbm', 'grade iv astrocytoma'],
                'lymphoma': ['non-hodgkin lymphoma', 'hodgkin lymphoma', 'diffuse large b-cell lymphoma'],
                'leukemia': ['acute myeloid leukemia', 'aml', 'chronic lymphocytic leukemia', 'cll']
            }
        }
    
    def calculate_weighted_validity_score(self, issues: list) -> float:
        """
        Calculate validity score based on weighted severity of issues
        """
        issue_weights = {
            'wrong_cancer_type': 0.4,        # Critical medical error
            'biomarker_mismatch_high': 0.3,  # Major therapeutic mismatch (high score without alignment)
            'biomarker_mismatch_low': 0.2,   # Missed opportunity (low score with alignment)
            'cancer_type_low': 0.25,         # Low score despite cancer match
            'stage_mismatch': 0.15,          # Moderate issue
            'ecog_mismatch': 0.1             # Minor consideration
        }
        
        total_penalty = 0.0
        for issue_text in issues:
            # Categorize issue type based on text content
            if 'mismatched cancer type' in issue_text:
                total_penalty += issue_weights['wrong_cancer_type']
            elif 'without' in issue_text and 'targeted therapy' in issue_text:
                total_penalty += issue_weights['biomarker_mismatch_high']
            elif 'despite' in issue_text and 'alignment' in issue_text:
                total_penalty += issue_weights['biomarker_mismatch_low']
            elif 'despite cancer type match' in issue_text:
                total_penalty += issue_weights['cancer_type_low']
            elif 'Phase III' in issue_text or 'stage' in issue_text.lower():
                total_penalty += issue_weights['stage_mismatch']
            elif 'ECOG' in issue_text:
                total_penalty += issue_weights['ecog_mismatch']
            else:
                # Default penalty for unrecognized issues
                total_penalty += 0.2
        
        return max(0, 1 - total_penalty)
    
    def quick_evaluation_checklist(self, llm_result: dict) -> dict[str, bool]:
        """
        Quick checklist for evaluating a single LLM result
        """
        score = llm_result.get('score', 0)
        reasoning = llm_result.get('reasoning', '').lower()
        
        checklist = {
            '✅ Score in valid range (0-100)': 0 <= score <= 100,
            '✅ Reasoning not empty': len(reasoning.strip()) > 10,
            '✅ Mentions medical concepts': any(word in reasoning for word in [
                'biomarker', 'stage', 'cancer', 'therapy', 'treatment', 'phase'
            ]),
            '✅ Score-reasoning alignment': not (
                (score > 70 and any(word in reasoning for word in ['poor', 'wrong', 'inappropriate'])) or
                (score < 30 and any(word in reasoning for word in ['excellent', 'perfect', 'ideal']))
            ),
            '✅ Specific (not just "good match")': len(reasoning.split()) > 15
        }
        
        return checklist
    
    def evaluate_response_quality(self, llm_results: list[dict]) -> dict[str, Any]:
        """
        Evaluate the quality of LLM responses using quick checklist
        """
        all_checklists = []
        
        for result in llm_results:
            checklist = self.quick_evaluation_checklist(result)
            all_checklists.append({
                'nct_id': result.get('nct_id', 'Unknown'),
                'checklist': checklist,
                'passed_checks': sum(checklist.values()),
                'total_checks': len(checklist)
            })
        
        # Calculate overall quality metrics
        total_checks = sum(item['total_checks'] for item in all_checklists)
        passed_checks = sum(item['passed_checks'] for item in all_checklists)
        quality_score = passed_checks / total_checks if total_checks > 0 else 0
        
        return {
            'overall_quality_score': quality_score,
            'detailed_checklists': all_checklists,
            'summary': f"Response Quality: {quality_score:.2%} ({passed_checks}/{total_checks} checks passed)"
        }
    
    def evaluate_medical_validity(self, llm_results: list[dict], patient_data: dict, trial_data_list: list[dict]) -> dict[str, Any]:
        """
        Check if the medical reasoning makes sense
        
        Args:
            llm_results: list of results from match_patient_to_trials (with nct_id, score, reasoning)
            patient_data: Patient information dictionary (from Patient.to_dict())
            trial_data_list: list of trial data dictionaries from find_trials
        """
        validity_scores = []
        
        # Create mapping from NCT ID to trial data
        trial_lookup = {}
        for i, trial in enumerate(trial_data_list):
            nct_id = trial.get('NCT Number', f'trial_{i}')
            trial_lookup[nct_id] = trial
        
        for result in llm_results:
            score = result.get('score', 0)
            reasoning = result.get('reasoning', '').lower()
            nct_id = result.get('nct_id', '')
            
            # Get corresponding trial info
            trial_info = trial_lookup.get(nct_id, {})
            
            validity_issues = []
            
            # STEP 1: Check biomarker-drug alignment
            patient_biomarkers = patient_data.get('biomarkers', [])
            trial_interventions = str(trial_info.get('Interventions', '')).lower()
            trial_conditions = str(trial_info.get('Conditions', '')).lower()
            trial_title = str(trial_info.get('Brief Title', '')).lower()
            
            # Combine all trial text for drug matching
            trial_text = f"{trial_interventions} {trial_conditions} {trial_title}"
            
            biomarker_alignment_found = False
            for biomarker in patient_biomarkers:
                biomarker_clean = biomarker.strip().replace(' ', '').lower()
                if biomarker_clean in self.medical_knowledge_base['biomarker_drug_mappings']:
                    expected_drugs = self.medical_knowledge_base['biomarker_drug_mappings'][biomarker_clean]
                    drug_mentioned = any(drug.lower() in trial_text for drug in expected_drugs)
                    
                    if drug_mentioned:
                        biomarker_alignment_found = True
                        if score < 50:
                            validity_issues.append(f"Low score ({score}) despite {biomarker}-drug alignment")
                    elif not drug_mentioned and score > 70:
                        validity_issues.append(f"High score ({score}) without {biomarker}-targeted therapy")
            
            # STEP 2: Check cancer type matching
            patient_cancer = patient_data.get('cancer_type', '').lower()
            
            # Check if patient cancer type matches trial conditions
            cancer_match = patient_cancer in trial_conditions
            if not cancer_match:
                # Check synonyms
                synonyms = self.medical_knowledge_base['cancer_synonyms'].get(patient_cancer, [])
                cancer_match = any(synonym in trial_conditions for synonym in synonyms)
            
            if not cancer_match and score > 50:
                validity_issues.append(f"High score ({score}) for mismatched cancer type")
            elif cancer_match and score < 30:
                validity_issues.append(f"Low score ({score}) despite cancer type match")
            
            # STEP 3: Check stage appropriateness  
            patient_stage = patient_data.get('stage', '').lower()
            if 'iv' in patient_stage or 'stage 4' in patient_stage or 'metastatic' in patient_stage:
                # Advanced cancer - should prioritize later phase trials or experimental treatments
                if 'phase iii' in trial_title and score < 60:
                    validity_issues.append(f"Low score ({score}) for Phase III trial in advanced cancer")
            
            # STEP 4: Check ECOG status appropriateness
            ecog_status = patient_data.get('ecog_status', 0)
            if ecog_status >= 2:  # Poor performance status
                if score > 70:
                    validity_issues.append(f"High score ({score}) for poor ECOG status ({ecog_status})")
            
            # Score validity based on weighted severity of issues
            validity_score = self.calculate_weighted_validity_score(validity_issues)
            validity_scores.append({
                'nct_id': nct_id,
                'validity_score': validity_score,
                'issues': validity_issues,
                'biomarker_alignment': biomarker_alignment_found
            })
        
        overall_validity = statistics.mean([v['validity_score'] for v in validity_scores]) if validity_scores else 0
        
        return {
            'medical_validity_score': overall_validity,
            'detailed_results': validity_scores,
            'summary': f"Medical Validity: {overall_validity:.2%}",
            'total_issues': sum(len(v['issues']) for v in validity_scores)
        }


def evaluate_response_quality(llm_results: list[dict]) -> dict[str, Any]:
    """
    Convenience function to evaluate response quality of matching results
    
    Args:
        llm_results: Output from match_patient_to_trials function
    
    Returns:
        Response quality evaluation results
    """
    evaluator = TrialMatchingEvaluator()
    return evaluator.evaluate_response_quality(llm_results)


def evaluate_medical_validity(llm_results: list[dict], patient_data: dict, trial_data_list: list[dict]) -> dict[str, Any]:
    """
    Convenience function to evaluate medical validity of matching results
    
    Args:
        llm_results: Output from match_patient_to_trials function
        patient_data: Patient data dictionary (from Patient.to_dict())
        trial_data_list: list of trial data from find_trials
    
    Returns:
        Medical validity evaluation results
    """
    evaluator = TrialMatchingEvaluator()
    return evaluator.evaluate_medical_validity(llm_results, patient_data, trial_data_list)