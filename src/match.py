"""
Main file for matching patients to clinical trials
"""

from data.patient_data import load_patient_data, Patient
from data.trial_data import find_trials
import argparse
import asyncio
import json
from prompts.generate_prompts import generate_patient_trial_criteria_prompts, generate_trial_matching_prompts
from llm.ollama import llm_response
from pprint import pprint

import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.eval_scripts import evaluate_medical_validity, evaluate_response_quality


def parse_llm_json_response(raw_response: str, trial_id: str = "unknown") -> dict:
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
            fixed_json = re.sub(r'("reasoning":\s*")([^"]*)"([^"]*)"([^"]*")', r'\1\2\"\3\"\4', fixed_json)
            fixed_json = re.sub(r'("confidence_reasoning":\s*")([^"]*)"([^"]*)"([^"]*")', r'\1\2\"\3\"\4', fixed_json)
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


def extract_patient_clinical_trial_criteria(patient_id: str) -> dict:
    """
    Extract clinical trial criteria from patient data
    """

    # Load patient data into Patient object
    patient_dict = load_patient_data(patient_id).to_dict()

    # Generate prompts for LLM to extract clinical trial criteria from patient data
    prompts = generate_patient_trial_criteria_prompts(patient_dict)

    patient_trial_criteria = llm_response(prompts['system_prompt'], prompts['user_prompt'])
    
    # Debug: Print the raw LLM response
    print(f"Raw LLM response: {patient_trial_criteria}")
    print(f"Response type: {type(patient_trial_criteria)}")
    print(f"Response length: {len(patient_trial_criteria) if patient_trial_criteria else 0}")

    # Use the helper function to parse JSON with fallback strategies
    parsed_response = parse_llm_json_response(patient_trial_criteria, "patient criteria")
    
    # If parsing failed completely, check if we got a default structure and enhance it
    if parsed_response.get("matching_score") == 0 and "Could not parse LLM response" in parsed_response.get("reasoning", ""):
        # This means it was a complete parsing failure, return patient criteria default
        return {
            "conditions": ["cancer"],
            "interventions": [],
            "trial_phases": ["PHASE1", "PHASE2", "PHASE3"],
            "inclusion_keywords": [],
            "exclusion_keywords": [],
            "geographic_region": "Unknown",
            "medical_context": {
                "treatment_line": "Unknown",
                "prognosis_category": "Unknown",
                "biomarker_strategy": "Unknown",
                "trial_priority": "Unknown",
                "special_considerations": "Unknown"
            }
        }
    
    return parsed_response

# Load trial data - list of trials for that condition

# Get LLM to embed and enhance trial data to extract relevant information

# Match patients to trials using LLM to score and rank trials
def match_patient_to_trials(patient_id: str) -> list:
    """
    Match a patient to trials using LLM to score and rank trials
    """
    patient_data = load_patient_data(patient_id)
    patient_trial_criteria = extract_patient_clinical_trial_criteria(patient_id)
    
    # Get trial data using the patient's specific cancer type
    print(f"Searching for trials for cancer type: {patient_data.cancer_type}")
    trial_data = asyncio.run(find_trials(patient_data.cancer_type))
    
    # For each trial, get the LLM to provide a score for how well the trial matches the patient's criteria
    # Provide a descending list of trials based on the score with the NCT number of the trial

    trial_scores = []

    for i, trial in enumerate(trial_data):
        # Prompt LLM to take in patient trial criteria and trial info and provide score
        print(f"Evaluating trial {i+1} of {len(trial_data)}")

        try:
            matching_prompts = generate_trial_matching_prompts(patient_trial_criteria, trial)
            raw_response = llm_response(matching_prompts['system_prompt'], matching_prompts['user_prompt'])
            
            # Use helper function to parse JSON response
            trial_evaluation = parse_llm_json_response(raw_response, f"trial {i+1}")

            trial_scores.append({
                'nct_id': trial.get('NCT Number', f'trial_{i+1}'),
                'score': trial_evaluation.get('matching_score', 0),
                'confidence_score': trial_evaluation.get('confidence_score', 0),
                'reasoning': trial_evaluation.get('reasoning', 'No reasoning provided'),
                'confidence_reasoning': trial_evaluation.get('confidence_reasoning', 'No confidence reasoning provided')
            })
        except Exception as e:
            print(f"Error evaluating trial {i+1}: {e}")
            trial_scores.append({
                'nct_id': trial.get('NCT Number', f'trial_{i+1}'),
                'score': 0,
                'confidence_score': 0,
                'reasoning': f'Error during evaluation: {str(e)}',
                'confidence_reasoning': 'No confidence assessment due to error'
            })

    # Sort trials by score in descending order
    trial_scores.sort(key=lambda x: x['score'], reverse=True)

    print(f"\nCompleted evaluation. Found {len(trial_scores)} scored trials.")

    return trial_scores


# Evaluate matches using medical validity and response quality evaluation
def evaluate_matches(trial_scores: list, patient_data: dict, trial_data: list) -> dict:
    """
    Evaluate matches using both medical validity and response quality evaluation
    """
    
    # Evaluate medical validity
    validity_results = evaluate_medical_validity(trial_scores, patient_data, trial_data)
    
    # Evaluate response quality
    quality_results = evaluate_response_quality(trial_scores)
    
    # Print summaries
    print(f"\n{validity_results['summary']}")
    print(f"Total validity issues found: {validity_results['total_issues']}")
    print(f"{quality_results['summary']}")
    
    # Combine results
    combined_results = {
        'medical_validity': validity_results,
        'response_quality': quality_results,
        'overall_summary': {
            'validity_score': validity_results['medical_validity_score'],
            'quality_score': quality_results['overall_quality_score'],
            'combined_score': (validity_results['medical_validity_score'] + quality_results['overall_quality_score']) / 2
        }
    }
    
    return combined_results

# Return results in ranked list of trials for that patient


def main():
    parser = argparse.ArgumentParser(description='Match patients to clinical trials')
    parser.add_argument('--patient_id', type=str, required=True, help='The ID of the patient to match')
    args = parser.parse_args()

    # Get patient data and trial data for evaluation
    patient_data = load_patient_data(args.patient_id)
    trial_data = asyncio.run(find_trials(patient_data.cancer_type))
    
    # Match patient to trials
    trial_scores = match_patient_to_trials(args.patient_id)
    pprint(trial_scores)
    
    # Evaluate the matches for medical validity and response quality
    evaluation = evaluate_matches(trial_scores, patient_data.to_dict(), trial_data)
    
    # Print overall combined score
    overall = evaluation['overall_summary']
    print(f"\n=== OVERALL EVALUATION SUMMARY ===")
    print(f"Medical Validity: {overall['validity_score']:.1%}")
    print(f"Response Quality: {overall['quality_score']:.1%}")
    print(f"Combined Score: {overall['combined_score']:.1%}")
    
    # Print detailed evaluation results
    print("\n=== DETAILED EVALUATION RESULTS ===")
    validity_results = evaluation['medical_validity']['detailed_results']
    quality_results = evaluation['response_quality']['detailed_checklists']
    
    for i, result in enumerate(validity_results):
        trial_info = trial_scores[i] if i < len(trial_scores) else {}
        quality_info = quality_results[i] if i < len(quality_results) else {}
        
        print(f"Trial {result['nct_id']}:")
        print(f"  Matching Score: {trial_info.get('score', 'N/A')}")
        print(f"  Confidence Score: {trial_info.get('confidence_score', 'N/A')}")
        print(f"  Validity Score: {result['validity_score']:.2%}")
        print(f"  Quality Score: {quality_info.get('passed_checks', 0)}/{quality_info.get('total_checks', 5)} checks passed")
        
        if result['issues']:
            print(f"  Validity Issues: {', '.join(result['issues'])}")
        print(f"  Biomarker alignment: {'Yes' if result['biomarker_alignment'] else 'No'}")
        
        # Show failed quality checks
        if quality_info.get('checklist'):
            failed_checks = [check for check, passed in quality_info['checklist'].items() if not passed]
            if failed_checks:
                print(f"  Failed Quality Checks: {', '.join(failed_checks)}")
        
        if trial_info.get('confidence_reasoning'):
            print(f"  Confidence reasoning: {trial_info.get('confidence_reasoning', 'N/A')}")
        print()

    # patient_data = load_patient_data(args.patient_id)
    # print(patient_data)

    # clinical_trial_criteria = extract_patient_clinical_trial_criteria(args.patient_id)
    # print(clinical_trial_criteria)

    # try:
    #     # Get trial data as JSON
    #     trial_data = asyncio.run(find_trials('cancer'))
    #     print(f"Raw trial data type: {type(trial_data)}")
        
    #     if trial_data:
    #         # Parse the JSON response
    #         try:
    #             print(f"Parsed JSON data type: {type(trial_data)}")

    #             print("=" * 50)
    #             print("Trial data:")

    #             print(trial_data)

    #             print("=" * 50)

    #             print(f"Number of trials found: {len(trial_data)}")
                
    #             # Print all trials
    #             for i, trial in enumerate(trial_data):
    #                 print(f"\n=== TRIAL {i+1} ===")
    #                 if isinstance(trial, dict):
    #                     print(json.dumps(trial, indent=2))
    #                 else:
    #                     print(str(trial))
    #                 print("=" * 50)
                    
    #         except json.JSONDecodeError as e:
    #             print(f"Error parsing JSON: {e}")
    #             print(f"Raw response: {trial_data_json[:500]}...")
    #     else:
    #         print("No trial data returned")
    # except Exception as e:
    #     print(f"Error getting trial data: {e}")
    #     import traceback
    #     traceback.print_exc()


if __name__ == '__main__':
    main()