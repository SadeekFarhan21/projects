from typing import List, Dict, Any, Optional
import requests


class ClinicalTrialsService:
    """
    Service to search and fetch clinical trials from ClinicalTrials.gov API v2
    """
    
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    def search_trials(
        self,
        condition: str,
        max_results: int = 20,
        phases: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search clinical trials by condition
        """
        params = {
            "query.cond": condition,
            "pageSize": min(max_results, 100),
            "format": "json",
        }
        
        # Add phase filter if provided
        if phases:
            params["query.phase"] = ",".join(phases)
        
        # Add status filter if provided  
        if statuses:
            params["query.status"] = ",".join(statuses)
        
        try:
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            studies = data.get("studies", [])
            return [self._parse_study(study) for study in studies]
        except Exception as e:
            print(f"ClinicalTrials.gov search error: {e}")
            return []
    
    def _parse_study(self, study: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a study record into simplified format"""
        try:
            protocol = study.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            descr_module = protocol.get("descriptionModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            interventions_module = protocol.get("armsInterventionsModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            design_module = protocol.get("designModule", {})
            
            # Extract NCT ID
            nct_id = id_module.get("nctId", "")
            
            # Title
            title = id_module.get("briefTitle", "")
            
            # Conditions
            conditions = conditions_module.get("conditions", [])
            condition_str = ", ".join(conditions) if conditions else ""
            
            # Interventions
            interventions_list = interventions_module.get("interventions", [])
            intervention_names = []
            for intervention in interventions_list:
                name = intervention.get("name", "")
                if name:
                    intervention_names.append(name)
            
            # Phase
            phases = design_module.get("phases", [])
            phase = phases[0] if phases else "N/A"
            
            # Status
            status = status_module.get("overallStatus", "")
            
            # Sponsor
            lead_sponsor = sponsor_module.get("leadSponsor", {})
            sponsor = lead_sponsor.get("name", "")
            
            # Summary
            brief_summary = descr_module.get("briefSummary", "")
            
            return {
                "nct_id": nct_id,
                "title": title,
                "condition": condition_str,
                "interventions": intervention_names,
                "phase": phase,
                "status": status,
                "sponsor": sponsor,
                "brief_summary": brief_summary,
                "url": f"https://clinicaltrials.gov/study/{nct_id}"
            }
        except Exception as e:
            print(f"Error parsing study: {e}")
            return {
                "nct_id": "",
                "title": "Parse error",
                "condition": "",
                "interventions": [],
                "phase": "",
                "status": "",
                "sponsor": "",
                "brief_summary": "",
                "url": ""
            }

