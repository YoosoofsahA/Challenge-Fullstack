"""
Synthetic test for LLM-based patient to trial matching results
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import statistics
import sys
import os
import asyncio
import json

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts.generate_prompts import generate_patient_trial_criteria_prompts, generate_trial_matching_prompts
from src.llm.ollama import llm_response
from src.models.patient import Patient
from tests.eval_scripts import TrialMatchingEvaluator


def parse_llm_json_response_synthetic(raw_response: str, trial_id: str = "unknown") -> dict:
    """
    Helper function to parse LLM JSON responses with multiple fallback strategies
    
    Args:
        raw_response: Raw string response from LLM
        trial_id: Identifier for logging purposes
        
    Returns:
        Dictionary with parsed JSON data or default structure
    """
    # Strategy 1: Direct JSON parsing
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass
    
    print(f"JSON decode error for {trial_id}. Trying fallback strategies...")
    
    # Strategy 2: Extract JSON from markdown blocks or extra text
    import re
    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    if json_match:
        try:
            cleaned_json = json_match.group(0)
            result = json.loads(cleaned_json)
            print(f"   Successfully parsed {trial_id} with strategy 2")
            return result
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Fix common quote escaping issues
    if json_match:
        try:
            fixed_json = json_match.group(0)
            # Fix quotes inside string values
            fixed_json = re.sub(r'(\"reasoning\":\s*")([^"]*)\"([^"]*)\"([^"]*")', r'\1\2\"\3\"\4', fixed_json)
            fixed_json = re.sub(r'(\"confidence_reasoning\":\s*")([^"]*)\"([^"]*)\"([^"]*")', r'\1\2\"\3\"\4', fixed_json)
            result = json.loads(fixed_json)
            print(f"   Successfully parsed {trial_id} with strategy 3")
            return result
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Manual regex extraction
    try:
        score_match = re.search(r'"matching_score":\s*(\d+)', raw_response)
        confidence_match = re.search(r'"confidence_score":\s*(\d+)', raw_response)
        reasoning_match = re.search(r'"reasoning":\s*"([^"]*(?:"[^"]*)*)"', raw_response)
        conf_reasoning_match = re.search(r'"confidence_reasoning":\s*"([^"]*(?:"[^"]*)*)"', raw_response)
        
        result = {
            "matching_score": int(score_match.group(1)) if score_match else 0,
            "confidence_score": int(confidence_match.group(1)) if confidence_match else 0,
            "reasoning": reasoning_match.group(1) if reasoning_match else "Could not extract reasoning",
            "confidence_reasoning": conf_reasoning_match.group(1) if conf_reasoning_match else "Could not extract confidence reasoning"
        }
        print(f"   Successfully parsed {trial_id} with strategy 4 (manual extraction)")
        return result
    except Exception:
        pass
    
    # Final fallback: Return default structure
    print(f"   Could not parse {trial_id} response with any strategy")
    return {
        "matching_score": 0,
        "confidence_score": 0,
        "reasoning": f"Could not parse LLM response: {raw_response[:200]}...",
        "confidence_reasoning": "No confidence assessment due to parsing error"
    }


class SyntheticPatient(Patient):
    """Synthetic patient inheriting from the real Patient class"""
    
    def __init__(self, patient_id: str, age: int, gender: str, cancer_type: str, 
                 stage: str, biomarkers: List[str], location: str, ecog_status: int, 
                 prior_treatments: List[str]):
        # Convert lists to semicolon/comma separated strings as expected by Patient class
        biomarkers_str = ';'.join(biomarkers) if biomarkers else ''
        prior_treatments_str = ','.join(prior_treatments) if prior_treatments else ''
        
        # Initialize parent Patient class
        super().__init__(
            patient_id=patient_id,
            age=age,
            gender=gender,
            cancer_type=cancer_type,
            stage=stage,
            biomarkers=biomarkers_str,
            location=location,
            ecog_status=ecog_status,
            prior_treatments=prior_treatments_str
        )


class SyntheticTrial:
    """Synthetic trial following the BioMCP trial data structure"""
    def __init__(self, nct_number: str, title: str, conditions: str, interventions: str, 
                 phases: str, study_status: str = "Recruiting", enrollment: int = 100):
        self.data = {
            'NCT Number': nct_number,
            'Study Title': title,
            'Brief Title': title,
            'Conditions': conditions,
            'Interventions': interventions,
            'Phases': phases,
            'Study Status': study_status,
            'Study Type': 'Interventional',
            'Enrollment': enrollment,
            'Brief Summary': f"A clinical trial studying {interventions} for {conditions}"
        }
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def to_dict(self) -> dict:
        return self.data.copy()


class RuleBasedMatcher:
    """Rule-based matching system for ground truth comparison"""
    
    def __init__(self):
        self.biomarker_drug_mapping = {
            'her2+': ['trastuzumab', 'pertuzumab', 'ado-trastuzumab', 'fam-trastuzumab'],
            'er+': ['tamoxifen', 'letrozole', 'anastrozole', 'fulvestrant'],
            'egfr+': ['erlotinib', 'gefitinib', 'osimertinib', 'afatinib'],
            'brafv600e': ['vemurafenib', 'dabrafenib', 'encorafenib'],
            'alk+': ['crizotinib', 'alectinib', 'ceritinib', 'brigatinib'],
            'pd-l1+': ['pembrolizumab', 'nivolumab', 'atezolizumab', 'durvalumab']
        }
        
        self.cancer_synonyms = {
            'breast cancer': ['breast carcinoma', 'mammary carcinoma', 'breast neoplasm'],
            'lung cancer': ['lung carcinoma', 'nsclc', 'non-small cell lung cancer'],
            'colorectal cancer': ['colon cancer', 'rectal cancer', 'crc'],
            'melanoma': ['malignant melanoma', 'cutaneous melanoma']
        }
    
    def calculate_match_score(self, patient: SyntheticPatient, trial: SyntheticTrial) -> Dict[str, Any]:
        """Calculate rule-based match score for patient-trial pair"""
        score = 0
        reasoning_parts = []
        
        # 1. Cancer type matching (40 points)
        cancer_match = self._check_cancer_match(patient.cancer_type, trial.get('Conditions', ''))
        if cancer_match:
            score += 40
            reasoning_parts.append(f"Cancer type match: {patient.cancer_type}")
        else:
            reasoning_parts.append(f"Cancer type mismatch: {patient.cancer_type} vs {trial.get('Conditions')}")
        
        # 2. Biomarker-drug alignment (30 points)
        biomarker_score, biomarker_reasoning = self._check_biomarker_alignment(
            patient.biomarkers, trial.get('Interventions', '')
        )
        score += biomarker_score
        reasoning_parts.append(biomarker_reasoning)
        
        # 3. Age eligibility (10 points)
        if 18 <= patient.age <= 75:  # Standard eligibility range
            score += 10
            reasoning_parts.append("Age within typical eligibility range")
        else:
            reasoning_parts.append("Age may be outside eligibility range")
        
        # 4. Performance status (10 points)
        if patient.ecog_status <= 1:
            score += 10
            reasoning_parts.append("Good performance status (ECOG ≤1)")
        elif patient.ecog_status == 2:
            score += 5
            reasoning_parts.append("Moderate performance status (ECOG 2)")
        else:
            reasoning_parts.append("Poor performance status (ECOG ≥3)")
        
        # 5. Stage appropriateness (10 points)
        stage_score = self._check_stage_appropriateness(patient.stage, trial.get('Phases', ''))
        score += stage_score
        if stage_score > 5:
            reasoning_parts.append("Trial phase appropriate for disease stage")
        else:
            reasoning_parts.append("Trial phase may not be optimal for disease stage")
        
        return {
            'nct_id': trial.get('NCT Number'),
            'rule_based_score': min(100, score),  # Cap at 100
            'reasoning': '; '.join(reasoning_parts),
            'cancer_match': cancer_match,
            'biomarker_score': biomarker_score
        }
    
    def _check_cancer_match(self, patient_cancer: str, trial_conditions: str) -> bool:
        """Check if patient cancer type matches trial conditions"""
        patient_cancer = patient_cancer.lower()
        trial_conditions = trial_conditions.lower()
        
        # Direct match
        if patient_cancer in trial_conditions:
            return True
        
        # Check synonyms
        synonyms = self.cancer_synonyms.get(patient_cancer, [])
        return any(synonym in trial_conditions for synonym in synonyms)
    
    def _check_biomarker_alignment(self, patient_biomarkers: List[str], trial_interventions: str) -> tuple:
        """Check biomarker-drug alignment and return score and reasoning"""
        if not patient_biomarkers:
            return 0, "No biomarkers specified"
        
        trial_interventions = trial_interventions.lower()
        total_biomarker_score = 0
        matched_biomarkers = []
        
        for biomarker in patient_biomarkers:
            biomarker_clean = biomarker.strip().replace(' ', '').lower()
            if biomarker_clean in self.biomarker_drug_mapping:
                expected_drugs = self.biomarker_drug_mapping[biomarker_clean]
                if any(drug.lower() in trial_interventions for drug in expected_drugs):
                    total_biomarker_score += 30 / len(patient_biomarkers)  # Distribute 30 points across biomarkers
                    matched_biomarkers.append(biomarker)
        
        if matched_biomarkers:
            reasoning = f"Biomarker alignment found: {', '.join(matched_biomarkers)}"
        else:
            reasoning = "No biomarker-drug alignment detected"
        
        return int(total_biomarker_score), reasoning
    
    def _check_stage_appropriateness(self, patient_stage: str, trial_phases: str) -> int:
        """Check if trial phase is appropriate for patient's disease stage"""
        stage = patient_stage.lower()
        phases = trial_phases.lower()
        
        # Advanced stage patients (Stage III/IV) - prefer Phase I/II trials
        if any(advanced in stage for advanced in ['iii', 'iv', '3', '4', 'metastatic', 'advanced']):
            if any(phase in phases for phase in ['phase 1', 'phase i', 'phase 2', 'phase ii']):
                return 10
            elif any(phase in phases for phase in ['phase 3', 'phase iii']):
                return 7  # Still reasonable
        
        # Early stage patients (Stage I/II) - prefer Phase II/III trials
        elif any(early in stage for early in ['i', 'ii', '1', '2', 'early']):
            if any(phase in phases for phase in ['phase 2', 'phase ii', 'phase 3', 'phase iii']):
                return 10
            elif any(phase in phases for phase in ['phase 1', 'phase i']):
                return 5  # Less optimal
        
        return 5  # Default moderate score


def create_synthetic_patients() -> List[SyntheticPatient]:
    """Create a set of synthetic patients for testing"""
    return [
        # Perfect HER2+ breast cancer match case
        SyntheticPatient(
            patient_id="SYN001",
            age=52,
            gender="Female",
            cancer_type="Breast Cancer",
            stage="Stage II",
            biomarkers=["HER2+", "ER+"],
            location="Boston, MA",
            ecog_status=0,
            prior_treatments=["Surgery"]
        ),
        
        # Advanced lung cancer case
        SyntheticPatient(
            patient_id="SYN002",
            age=68,
            gender="Male",
            cancer_type="Lung Cancer",
            stage="Stage IV",
            biomarkers=["EGFR+"],
            location="New York, NY",
            ecog_status=1,
            prior_treatments=["Chemotherapy", "Radiation"]
        ),
        
        # Young melanoma patient
        SyntheticPatient(
            patient_id="SYN003",
            age=34,
            gender="Female",
            cancer_type="Melanoma",
            stage="Stage III",
            biomarkers=["BRAF V600E"],
            location="San Francisco, CA",
            ecog_status=0,
            prior_treatments=["Surgery", "Immunotherapy"]
        ),
        
        # Poor performance status case
        SyntheticPatient(
            patient_id="SYN004",
            age=75,
            gender="Male",
            cancer_type="Colorectal Cancer",
            stage="Stage IV",
            biomarkers=[],
            location="Chicago, IL",
            ecog_status=3,
            prior_treatments=["Chemotherapy", "Surgery", "Radiation"]
        )
    ]


def create_synthetic_trials() -> List[SyntheticTrial]:
    """Create a set of synthetic trials for testing"""
    return [
        # Perfect match for HER2+ breast cancer patient
        SyntheticTrial(
            nct_number="NCT-SYN001",
            title="Trastuzumab and Pertuzumab for HER2+ Breast Cancer",
            conditions="HER2-positive Breast Cancer",
            interventions="Trastuzumab; Pertuzumab; Paclitaxel",
            phases="Phase 2"
        ),
        
        # Good match for lung cancer patient
        SyntheticTrial(
            nct_number="NCT-SYN002", 
            title="Osimertinib for EGFR+ Advanced NSCLC",
            conditions="Non-Small Cell Lung Cancer; EGFR-positive",
            interventions="Osimertinib",
            phases="Phase 1/2"
        ),
        
        # Moderate match - right cancer, wrong biomarker
        SyntheticTrial(
            nct_number="NCT-SYN003",
            title="Immunotherapy for Advanced Melanoma",
            conditions="Malignant Melanoma; Advanced Melanoma",
            interventions="Pembrolizumab; Nivolumab",
            phases="Phase 3"
        ),
        
        # Poor match - wrong cancer type
        SyntheticTrial(
            nct_number="NCT-SYN004",
            title="Prostate Cancer Hormone Therapy Study",
            conditions="Prostate Cancer; Castration-Resistant Prostate Cancer", 
            interventions="Abiraterone; Enzalutamide",
            phases="Phase 3"
        ),
        
        # Generic cancer trial (moderate match for most)
        SyntheticTrial(
            nct_number="NCT-SYN005",
            title="Pan-Cancer Immunotherapy Trial",
            conditions="Solid Tumors; Advanced Cancer",
            interventions="Pembrolizumab; Combination Therapy",
            phases="Phase 1"
        )
    ]


def run_synthetic_evaluation() -> Dict[str, Any]:
    """Run synthetic evaluation comparing rule-based vs LLM scoring"""
    patients = create_synthetic_patients()
    trials = create_synthetic_trials()
    matcher = RuleBasedMatcher()
    
    results = []
    
    print("=== SYNTHETIC EVALUATION RESULTS ===\n")
    
    for patient in patients:
        print(f"Patient {patient.patient_id}: {patient.cancer_type}, {patient.stage}, Biomarkers: {patient.biomarkers}")
        patient_results = []
        
        for trial in trials:
            rule_result = matcher.calculate_match_score(patient, trial)
            patient_results.append(rule_result)
        
        # Sort by rule-based score (ground truth ranking)
        patient_results.sort(key=lambda x: x['rule_based_score'], reverse=True)
        
        print("\nRule-based Rankings:")
        for i, result in enumerate(patient_results, 1):
            print(f"  {i}. {result['nct_id']} - Score: {result['rule_based_score']}")
            print(f"     Reasoning: {result['reasoning']}")
        
        results.append({
            'patient_id': patient.patient_id,
            'patient_data': patient.to_dict(),
            'rankings': patient_results
        })
        print("\n" + "="*80 + "\n")
    
    return {
        'synthetic_results': results,
        'patients': [p.to_dict() for p in patients],
        'trials': [t.to_dict() for t in trials]
    }


def compare_llm_vs_rule_based(llm_results: List[Dict], rule_based_results: List[Dict]) -> Dict[str, Any]:
    """Compare LLM rankings against rule-based ground truth"""
    if len(llm_results) != len(rule_based_results):
        return {"error": "Mismatched result lengths"}
    
    ranking_differences = []
    
    for llm_result, rule_result in zip(llm_results, rule_based_results):
        llm_score = llm_result.get('score', 0)
        rule_score = rule_result.get('rule_based_score', 0)
        
        # Simple correlation calculation
        score_diff = abs(llm_score - rule_score)
        ranking_differences.append(score_diff)
    
    avg_difference = statistics.mean(ranking_differences) if ranking_differences else 0
    
    # Calculate agreement categories
    close_agreement = sum(1 for diff in ranking_differences if diff <= 15)  # Within 15 points
    moderate_agreement = sum(1 for diff in ranking_differences if 15 < diff <= 30)
    poor_agreement = sum(1 for diff in ranking_differences if diff > 30)
    
    return {
        'average_score_difference': avg_difference,
        'close_agreement': close_agreement,
        'moderate_agreement': moderate_agreement,
        'poor_agreement': poor_agreement,
        'total_comparisons': len(ranking_differences),
        'agreement_rate': close_agreement / len(ranking_differences) if ranking_differences else 0
    }


def extract_patient_criteria_synthetic(patient: SyntheticPatient) -> dict:
    """Extract clinical trial criteria from synthetic patient data using LLM"""
    # Use the inherited to_dict method from Patient class
    patient_dict = patient.to_dict()
    prompts = generate_patient_trial_criteria_prompts(patient_dict)
    
    raw_response = llm_response(prompts['system_prompt'], prompts['user_prompt'])
    
    # Use the robust JSON parser
    parsed_response = parse_llm_json_response_synthetic(raw_response, f"patient {patient.patient_id}")
    
    # If parsing failed completely, check if we got a default structure and enhance it
    if parsed_response.get("matching_score") == 0 and "Could not parse LLM response" in parsed_response.get("reasoning", ""):
        # This means it was a complete parsing failure, return patient criteria default
        return {
            "conditions": [patient.cancer_type],
            "interventions": [],
            "trial_phases": ["PHASE1", "PHASE2", "PHASE3"],
            "inclusion_keywords": [],
            "exclusion_keywords": [],
            "geographic_region": patient.location,
            "medical_context": {
                "treatment_line": "Unknown",
                "prognosis_category": patient.stage,
                "biomarker_strategy": str(patient.biomarkers),
                "trial_priority": "Unknown",
                "special_considerations": f"ECOG {patient.ecog_status}"
            }
        }
    
    return parsed_response


def match_synthetic_patient_to_trials(patient: SyntheticPatient, trials: List[SyntheticTrial]) -> List[Dict]:
    """Run LLM matching for synthetic patient against synthetic trials"""
    print(f"\n🤖 Running LLM matching for Patient {patient.patient_id}...")
    
    # Extract patient criteria using LLM
    patient_criteria = extract_patient_criteria_synthetic(patient)
    
    llm_results = []
    
    for i, trial in enumerate(trials):
        print(f"   Evaluating trial {i+1}/{len(trials)}: {trial.get('NCT Number')}")
        
        try:
            matching_prompts = generate_trial_matching_prompts(patient_criteria, trial.to_dict())
            raw_response = llm_response(matching_prompts['system_prompt'], matching_prompts['user_prompt'])
            
            # Use the robust JSON parser
            trial_evaluation = parse_llm_json_response_synthetic(raw_response, trial.get('NCT Number'))
            
            llm_results.append({
                'nct_id': trial.get('NCT Number'),
                'score': trial_evaluation.get('matching_score', 0),
                'confidence_score': trial_evaluation.get('confidence_score', 0),
                'reasoning': trial_evaluation.get('reasoning', 'No reasoning provided'),
                'confidence_reasoning': trial_evaluation.get('confidence_reasoning', 'No confidence reasoning provided')
            })
            
        except Exception as e:
            print(f"      Error evaluating trial: {e}")
            llm_results.append({
                'nct_id': trial.get('NCT Number'),
                'score': 0,
                'confidence_score': 0,
                'reasoning': f'Error during evaluation: {str(e)}',
                'confidence_reasoning': 'No confidence assessment due to error'
            })
    
    # Sort by score in descending order
    llm_results.sort(key=lambda x: x['score'], reverse=True)
    return llm_results


def run_comprehensive_synthetic_evaluation() -> Dict[str, Any]:
    """Run both rule-based and LLM evaluations, then compare them"""
    patients = create_synthetic_patients()
    trials = create_synthetic_trials()
    matcher = RuleBasedMatcher()
    evaluator = TrialMatchingEvaluator()
    
    results = {
        'patients': [],
        'rule_based_results': [],
        'llm_results': [],
        'comparisons': [],
        'overall_comparison': {}
    }
    
    print("=== COMPREHENSIVE SYNTHETIC EVALUATION ===\n")
    
    for patient in patients:
        print(f"📋 Processing Patient {patient.patient_id}: {patient.cancer_type}, {patient.stage}")
        
        # Rule-based evaluation
        print("   🎯 Running rule-based matching...")
        rule_based_rankings = []
        for trial in trials:
            rule_result = matcher.calculate_match_score(patient, trial)
            rule_based_rankings.append(rule_result)
        rule_based_rankings.sort(key=lambda x: x['rule_based_score'], reverse=True)
        
        # LLM evaluation
        llm_rankings = match_synthetic_patient_to_trials(patient, trials)
        
        # Compare results
        comparison = compare_llm_vs_rule_based(llm_rankings, rule_based_rankings)
        
        # Evaluate LLM response quality and medical validity
        quality_eval = evaluator.evaluate_response_quality(llm_rankings)
        validity_eval = evaluator.evaluate_medical_validity(
            llm_rankings, 
            patient.to_dict(), 
            [trial.to_dict() for trial in trials]
        )
        
        results['patients'].append(patient.to_dict())
        results['rule_based_results'].append(rule_based_rankings)
        results['llm_results'].append(llm_rankings)
        results['comparisons'].append({
            'patient_id': patient.patient_id,
            'score_comparison': comparison,
            'quality_evaluation': quality_eval,
            'validity_evaluation': validity_eval
        })
        
        print(f"   ✅ Completed evaluation for {patient.patient_id}")
        print(f"      Agreement rate: {comparison['agreement_rate']:.1%}")
        print(f"      Quality score: {quality_eval['overall_quality_score']:.1%}")
        print(f"      Validity score: {validity_eval['medical_validity_score']:.1%}")
        print()
    
    # Calculate overall metrics
    all_agreement_rates = [comp['score_comparison']['agreement_rate'] for comp in results['comparisons']]
    all_quality_scores = [comp['quality_evaluation']['overall_quality_score'] for comp in results['comparisons']]
    all_validity_scores = [comp['validity_evaluation']['medical_validity_score'] for comp in results['comparisons']]
    
    results['overall_comparison'] = {
        'average_agreement_rate': statistics.mean(all_agreement_rates),
        'average_quality_score': statistics.mean(all_quality_scores),
        'average_validity_score': statistics.mean(all_validity_scores),
        'total_patients': len(patients),
        'total_trials': len(trials),
        'total_comparisons': len(patients) * len(trials)
    }
    
    return results


if __name__ == "__main__":
    # Run comprehensive evaluation (rule-based + LLM + comparison)
    comprehensive_results = run_comprehensive_synthetic_evaluation()
    
    print("\n" + "="*100)
    print("=== DETAILED COMPARISON ANALYSIS ===")
    
    # Display results for each patient
    for i, patient_data in enumerate(comprehensive_results['patients']):
        patient_id = patient_data['patient_id']
        rule_based_rankings = comprehensive_results['rule_based_results'][i]
        llm_rankings = comprehensive_results['llm_results'][i]
        comparison = comprehensive_results['comparisons'][i]
        
        print(f"\n📋 Patient {patient_id} Results:")
        print(f"   Profile: {patient_data['cancer_type']}, {patient_data['stage']}")
        print(f"   Biomarkers: {patient_data['biomarkers']}, ECOG: {patient_data['ecog_status']}")
        
        # Show top 3 comparisons
        print(f"\n   📊 Top 3 Trial Comparisons (Rule-Based vs LLM):")
        for j in range(min(3, len(rule_based_rankings))):
            rule_result = rule_based_rankings[j]
            # Find corresponding LLM result
            llm_result = next((r for r in llm_rankings if r['nct_id'] == rule_result['nct_id']), None)
            
            if llm_result:
                score_diff = abs(rule_result['rule_based_score'] - llm_result['score'])
                agreement = "✅" if score_diff <= 15 else "⚠️" if score_diff <= 30 else "❌"
                
                print(f"      {j+1}. {rule_result['nct_id']}")
                print(f"         Rule-Based: {rule_result['rule_based_score']}/100")
                print(f"         LLM Score:  {llm_result['score']}/100")
                print(f"         Confidence: {llm_result['confidence_score']}/100")
                print(f"         Agreement: {agreement} (diff: {score_diff} pts)")
                print(f"         Rule reasoning: {rule_result['reasoning'][:80]}...")
                print(f"         LLM reasoning:  {llm_result['reasoning'][:80]}...")
        
        # Show evaluation metrics
        score_comp = comparison['score_comparison']
        quality_eval = comparison['quality_evaluation']
        validity_eval = comparison['validity_evaluation']
        
        print(f"\n   📈 Evaluation Metrics:")
        print(f"      Agreement Rate: {score_comp['agreement_rate']:.1%} ({score_comp['close_agreement']}/{score_comp['total_comparisons']} close)")
        print(f"      Average Score Difference: {score_comp['average_score_difference']:.1f} points")
        print(f"      Response Quality: {quality_eval['overall_quality_score']:.1%}")
        print(f"      Medical Validity: {validity_eval['medical_validity_score']:.1%}")
        
        if validity_eval['total_issues'] > 0:
            print(f"      ⚠️ Medical Issues Found: {validity_eval['total_issues']}")
    
    # Overall summary
    overall = comprehensive_results['overall_comparison']
    print(f"\n" + "="*100)
    print("=== OVERALL EVALUATION SUMMARY ===")
    print(f"📊 Dataset: {overall['total_patients']} patients × {overall['total_trials']} trials = {overall['total_comparisons']} comparisons")
    print(f"🎯 Average Agreement Rate: {overall['average_agreement_rate']:.1%}")
    print(f"✅ Average Response Quality: {overall['average_quality_score']:.1%}")
    print(f"⚕️ Average Medical Validity: {overall['average_validity_score']:.1%}")
    
    # Performance interpretation
    print(f"\n🔍 Performance Interpretation:")
    
    # More nuanced evaluation considering medical validity and response quality
    if overall['average_validity_score'] >= 0.95 and overall['average_quality_score'] >= 0.95:
        if overall['average_agreement_rate'] >= 0.3:
            print("   ✅ EXCELLENT: High medical validity with strong clinical reasoning")
            print("       LLM demonstrates conservative, clinically appropriate scoring")
        else:
            print("   ✅ GOOD: Excellent medical validity but conservative scoring approach")
            print("       LLM prioritizes clinical safety over partial credit scoring")
    elif overall['average_validity_score'] >= 0.8:
        if overall['average_agreement_rate'] >= 0.6:
            print("   ✅ GOOD: LLM shows reasonable alignment with medical logic")
        elif overall['average_agreement_rate'] >= 0.4:
            print("   ⚠️ MODERATE: LLM has some alignment but needs prompt tuning")
        else:
            print("   ⚠️ MIXED: Good medical reasoning but scoring philosophy differs from rule-based")
    else:
        print("   ❌ POOR: LLM medical reasoning needs improvement")
    
    if overall['average_quality_score'] < 0.8:
        print("   ⚠️ Consider improving prompt clarity for better response quality")
    
    if overall['average_validity_score'] < 0.8:
        print("   ⚠️ Consider adding more medical context to prompts")
    
    print("\n" + "="*100)
    print("\n=== EVALUATION FRAMEWORK SUMMARY ===")
    
    # Display scoring breakdown
    print("\n📊 Rule-Based Scoring System (100 points total):")
    print("   • Cancer Type Match: 40 points (exact match or synonyms)")
    print("   • Biomarker-Drug Alignment: 30 points (targeted therapy compatibility)")
    print("   • Age Eligibility: 10 points (standard 18-75 range)")
    print("   • Performance Status: 10 points (ECOG 0-1 preferred)")
    print("   • Stage Appropriateness: 10 points (phase matching disease stage)")
    
    print("\n🔬 Test Cases Designed:")
    print("   • Perfect Match: HER2+ breast cancer → HER2-targeted trial")
    print("   • Good Match: EGFR+ lung cancer → EGFR inhibitor trial")
    print("   • Moderate Match: Right cancer type, wrong biomarker")
    print("   • Poor Match: Wrong cancer type entirely")
    print("   • Generic Match: Pan-cancer immunotherapy trial")
    
    print("\n🎯 LLM Validation Guidelines:")
    print("   • Close Agreement: LLM score within ±15 points of rule-based score")
    print("   • Moderate Agreement: LLM score within ±30 points")
    print("   • Poor Agreement: LLM score differs by >30 points")
    
    print("\n=== INTEGRATION INSTRUCTIONS ===")
    print("To compare LLM results against this ground truth:")
    print("1. Run your LLM matching system on the synthetic patients")
    print("2. Use compare_llm_vs_rule_based() function")
    print("3. Analyze agreement rates and identify systematic biases")
    print("4. Tune prompts based on cases where LLM significantly disagrees with medical logic")
    
    print(f"\n✅ Synthetic evaluation framework ready!")
    print(f"   📁 {len(comprehensive_results['patients'])} test patients created")
    print(f"   🔬 5 test trials created") 
    print(f"   📊 {len(comprehensive_results['patients']) * 5} patient-trial combinations evaluated")
    print(f"   🎯 Rule-based ground truth established for all combinations")