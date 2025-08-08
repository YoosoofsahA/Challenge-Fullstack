"""
Data loader for loading trial data from Biomcp
"""

from biomcp.trials.search import (  # type: ignore
    RecruitingStatus,
    TrialQuery,
    search_trials,
)
import json

# Find trial data for a given condition
async def find_trials(condition: str) -> list:
    """
    Find trial data for a given condition
    """
    
    query = TrialQuery(
        conditions=[condition],
        recruiting_status=RecruitingStatus.OPEN,
        page_size=10, # NOTE: Keeping small for testing
    )

    trials = await search_trials(query, output_json=True)

    # Parse the JSON response
    trials_data = json.loads(trials)
    
    return trials_data

# TODO: Make search function more directed to the patient's cancer type