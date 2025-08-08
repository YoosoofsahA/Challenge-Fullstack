"""
Generate prompts for the LLM.
"""

def generate_patient_trial_criteria_prompts(patient_dict: dict) -> dict:
    """
    Generate system and user prompts for extracting clinical trial search criteria
    from a patient profile using an LLM.
    
    Args:
        patient_dict: Dictionary with patient data, e.g.:
        {
            'patient_id': 'P001',
            'age': 45,
            'gender': 'Female', 
            'cancer_type': 'Breast Cancer',
            'stage': 'Stage II',
            'biomarkers': ['HER2+', 'ER+'],
            'location': 'Boston MA',
            'ecog_status': 0,
            'prior_treatments': ['Surgery', 'Chemotherapy']
        }
    
    Returns:
        dict: {'system_prompt': str, 'user_prompt': str}
    """
    
    system_prompt = """You are a clinical oncology expert specializing in matching cancer patients to appropriate clinical trials. Your task is to analyze a patient's medical profile and extract structured search criteria that will be used to find relevant recruiting clinical trials.

EXPERTISE AREAS:
- Cancer staging and prognosis
- Biomarker-targeted therapies  
- Treatment sequencing and resistance patterns
- Clinical trial design and eligibility criteria
- FDA-approved and investigational cancer drugs

EXTRACTION GUIDELINES:

1. CONDITIONS: Include the primary cancer type plus relevant synonyms, subtypes, and related terms that might appear in trial descriptions.

2. INTERVENTIONS: Based on biomarkers, cancer type, stage, and prior treatments, identify:
   - Targeted therapies (match biomarkers to specific drugs/drug classes)
   - Standard treatments appropriate for this cancer type/stage
   - Investigational approaches likely to be studied
   - Combination therapies
   - Consider treatment resistance from prior therapies

3. TRIAL_PHASES: Recommend appropriate trial phases based on:
   - Early stage (I-II): May include adjuvant/neoadjuvant trials (Phase 2-3)
   - Advanced stage (III-IV): Focus on treatment trials (Phase 1-2)
   - Treatment-naive vs heavily pretreated
   - Performance status

4. INCLUSION_KEYWORDS: Additional terms that indicate trial relevance (e.g., biomarker names, specific cancer subtypes, treatment settings)

5. EXCLUSION_KEYWORDS: Terms that indicate trial unsuitability (e.g., wrong age group, incompatible prior treatments, wrong cancer type)

6. GEOGRAPHIC_CONSIDERATIONS: Note patient location for potential geographic filtering

7. MEDICAL_CONTEXT: Provide reasoning for your recommendations to help with trial matching

OUTPUT FORMAT: Return a valid JSON object with the specified structure. Do not include any text outside the JSON."""

    user_prompt = f"""Extract clinical trial search criteria from this cancer patient profile:

PATIENT PROFILE:
- Patient ID: {patient_dict.get('patient_id', 'Unknown')}
- Age: {patient_dict.get('age', 'Unknown')}
- Gender: {patient_dict.get('gender', 'Unknown')}  
- Cancer Type: {patient_dict.get('cancer_type', 'Unknown')}
- Stage: {patient_dict.get('stage', 'Unknown')}
- Biomarkers: {patient_dict.get('biomarkers', 'Unknown')}
- Location: {patient_dict.get('location', 'Unknown')}
- ECOG Performance Status: {patient_dict.get('ecog_status', 'Unknown')}
- Prior Treatments: {patient_dict.get('prior_treatments', 'Unknown')}

Return a JSON object with this exact structure:

{{
  "conditions": [
    "Primary cancer type",
    "Relevant synonyms and subtypes",
    "Related disease terms"
  ],
  "interventions": [
    "Specific targeted drugs based on biomarkers",
    "Drug classes appropriate for this cancer",
    "Standard treatments for this stage",
    "Investigational approaches",
    "Combination therapies"
  ],
  "trial_phases": [
    "PHASE1", "PHASE2", "PHASE3"
  ],
  "inclusion_keywords": [
    "Biomarker terms",
    "Cancer subtype terms", 
    "Treatment setting terms",
    "Other relevant medical terms"
  ],
  "exclusion_keywords": [
    "Inappropriate age groups",
    "Wrong cancer types",
    "Incompatible treatments",
    "Other exclusionary terms"
  ],
  "geographic_region": "Patient location for geographic filtering",
  "medical_context": {{
    "treatment_line": "First-line/Second-line/Heavily pretreated",
    "prognosis_category": "Early stage/Locally advanced/Metastatic",
    "biomarker_strategy": "Explanation of targeted therapy rationale",
    "trial_priority": "Priority trial types based on patient profile",
    "special_considerations": "Any unique factors affecting trial selection"
  }}
}}

Focus on creating comprehensive search criteria that will help find the most relevant and appropriate clinical trials for this specific patient profile.

IMPORTANT:
- Return as valid JSON only, no additional text
- Do not include any text outside the JSON
- Do not include any other text or comments
- Do not include any other formatting or comments
- Do not include any other formatting or comments
"""

    return {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt
    }


def generate_trial_matching_prompts(patient_criteria: dict, trial_data: dict) -> dict:
    """
    Generate system and user prompts for LLM to score patient-trial matches
    
    Args:
        patient_criteria: Dictionary with extracted patient criteria from previous LLM step
        trial_data: Single trial dictionary from BioMCP with trial information
        
    Returns:
        dict: {'system_prompt': str, 'user_prompt': str}
    """
    
    system_prompt = """You are a senior clinical oncologist and clinical trial expert with extensive experience in patient-trial matching. Your role is to evaluate how well a specific clinical trial matches a cancer patient's profile and medical needs.

EXPERTISE AREAS:
- Cancer treatment protocols and sequencing
- Clinical trial eligibility criteria interpretation
- Biomarker-targeted therapy selection
- Patient safety and contraindication assessment
- Treatment line appropriateness
- Geographic and logistical feasibility

SCORING CRITERIA (0-100 scale):

EXCELLENT MATCH (90-100):
- Perfect biomarker alignment (e.g., HER2+ patient → HER2-targeted trial)
- Cancer type and stage exactly match inclusion criteria
- Patient meets all key eligibility requirements
- Treatment is appropriate for patient's treatment line
- Trial phase suitable for patient's prognosis
- No significant exclusionary factors

GOOD MATCH (70-89):
- Strong biomarker or cancer type alignment
- Patient meets most major eligibility criteria
- Minor eligibility concerns that could be addressed
- Appropriate treatment approach for patient's condition
- Some potential benefits despite minor mismatches

MODERATE MATCH (50-69):
- Partial relevance to patient's condition
- Some eligibility criteria met but significant gaps
- Treatment approach somewhat appropriate
- May require additional screening or evaluation
- Mixed benefit-risk profile

POOR MATCH (30-49):
- Limited relevance to patient's specific condition
- Major eligibility criteria not met
- Treatment approach not optimal for patient
- Significant safety concerns or contraindications
- Unlikely to provide meaningful benefit

VERY POOR MATCH (0-29):
- No relevant match to patient's condition
- Patient clearly ineligible based on major criteria
- Wrong cancer type, stage, or biomarker profile
- Serious safety contraindications
- No potential benefit for this patient

EVALUATION FACTORS:
1. Cancer Type Match: Exact match vs. related vs. unrelated
2. Biomarker Alignment: Targeted therapy relevance
3. Disease Stage Appropriateness: Early vs. advanced stage suitability
4. Age and Gender Eligibility: Within specified ranges
5. Prior Treatment Compatibility: Treatment line appropriateness
6. Performance Status (ECOG): Patient functional status requirements
7. Geographic Accessibility: Trial location vs. patient location
8. Trial Phase Appropriateness: Phase 1 (experimental) vs. Phase 3 (standard care)
9. Safety Profile: Contraindications and risk factors
10. Treatment Goals Alignment: Curative vs. palliative vs. preventive

OUTPUT REQUIREMENTS:
- Provide a single numerical matching score from 0-100
- Provide a confidence score (0-100) indicating how certain you are about the matching score
- Include the trial's NCT ID
- Give brief reasoning for both the matching score and confidence score
- Return as valid JSON only, no additional text

CONFIDENCE SCORE GUIDELINES (0-100):
- HIGH CONFIDENCE (80-100): Clear eligibility criteria, well-defined patient population, obvious biomarker match/mismatch
- MODERATE CONFIDENCE (60-79): Some criteria are clear but missing key details, partial information available
- LOW CONFIDENCE (40-59): Limited trial information, unclear eligibility criteria, ambiguous patient match
- VERY LOW CONFIDENCE (0-39): Insufficient information to make reliable assessment, major information gaps

REQUIRED JSON OUTPUT FORMAT:
{
  "matching_score": [integer from 0-100],
  "confidence_score": [integer from 0-100],
  "nct_id": "NCT Number from the trial",
  "reasoning": "Brief explanation of the matching score focusing on key matching or mismatching factors",
  "confidence_reasoning": "Brief explanation of why you have this level of confidence in your matching score"
}"""

    # Extract key patient information for the prompt
    patient_summary = f"""
PATIENT CRITERIA:
- Cancer Type: {patient_criteria.get('conditions', ['Unknown'])[0] if patient_criteria.get('conditions') else 'Unknown'}
- Target Biomarkers: {', '.join(patient_criteria.get('inclusion_keywords', []))}
- Preferred Interventions: {', '.join(patient_criteria.get('interventions', [])[:5])}  
- Suitable Trial Phases: {', '.join(patient_criteria.get('trial_phases', []))}
- Medical Context: {patient_criteria.get('medical_context', {}).get('biomarker_strategy', 'Not specified')}
- Treatment Line: {patient_criteria.get('medical_context', {}).get('treatment_line', 'Unknown')}
- Prognosis Category: {patient_criteria.get('medical_context', {}).get('prognosis_category', 'Unknown')}
- Geographic Location: {patient_criteria.get('geographic_region', 'Unknown')}
"""

    # Extract key trial information
    trial_summary = f"""
TRIAL INFORMATION:
- NCT ID: {trial_data.get('NCT Number', 'Unknown')}
- Title: {trial_data.get('Study Title', 'No title')}
- Status: {trial_data.get('Study Status', 'Unknown')}
- Conditions: {trial_data.get('Conditions', 'Not specified')}
- Interventions: {trial_data.get('Interventions', 'Not specified')}
- Phases: {trial_data.get('Phases', 'Not specified')}
- Study Type: {trial_data.get('Study Type', 'Unknown')}
- Enrollment: {trial_data.get('Enrollment', 'Unknown')} participants
- Brief Summary: {trial_data.get('Brief Summary', 'No summary available')[:500]}...
"""

    user_prompt = f"""Evaluate how well this clinical trial matches the patient's criteria and provide a matching score from 0-100.

{patient_summary}

{trial_summary}

Analyze the match considering:
1. Does the trial's condition match the patient's cancer type?
2. Do the trial interventions align with the patient's biomarker profile?
3. Is the trial phase appropriate for the patient's disease stage and treatment history?
4. Would the patient likely meet the trial's eligibility criteria?
5. Is this trial a medically sound treatment option for this patient?

Return ONLY a JSON object with this exact format:
{{
  "nct_id": "{trial_data.get('NCT Number', 'Unknown')}",
  "matching_score": [0-100 integer],
  "confidence_score": [0-100 integer],
  "reasoning": "Brief explanation of why this matching score was assigned, highlighting key matching or mismatching factors",
  "confidence_reasoning": "Brief explanation of your confidence level in this assessment"
}}

IMPORTANT:
- Return as valid JSON only, no additional text
- Do not include any text outside the JSON
- Do not include any other text or comments
- Escape all quotes in string values using \"
- If your reasoning contains quotes, use \" instead of "
- Ensure all JSON string values are properly quoted and escaped
"""

    return {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt
    }