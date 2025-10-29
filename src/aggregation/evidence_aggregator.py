"""Evidence aggregation with weighted fusion."""

import numpy as np
from typing import Dict, List, Any

class EvidenceAggregator:
    """Aggregate evidence from multiple modalities."""
    
    def __init__(self, weights: Dict = None):
        self.weights = weights or {
            'visual': 0.35,
            'clinical': 0.20,
            'lab': 0.30,
            'geographic': 0.10,
            'literature': 0.05
        }
    
    def aggregate(self, initial_prediction: Dict, retrieved_evidence: Dict) -> Dict:
        """
        Aggregate evidence to refine prediction.
        
        Args:
            initial_prediction: CNN prediction with probabilities
            retrieved_evidence: Retrieved evidence from KG
        
        Returns:
            Refined prediction with explanation
        """
        # Visual modality
        visual_support = self._analyze_visual_evidence(
            retrieved_evidence['visual'],
            initial_prediction['class']
        )
        
        # Clinical modality  
        clinical_support = self._analyze_clinical_evidence(
            retrieved_evidence['clinical'],
            initial_prediction['class']
        )
        
        # Lab modality
        lab_support = self._analyze_lab_evidence(
            retrieved_evidence['lab'],
            initial_prediction['class']
        )
        
        # Geographic modality
        geo_support = self._analyze_geographic_evidence(
            retrieved_evidence['geographic'],
            initial_prediction['class']
        )
        
        # Weighted fusion
        evidence_scores = {
            'visual': visual_support * self.weights['visual'],
            'clinical': clinical_support * self.weights['clinical'],
            'lab': lab_support * self.weights['lab'],
            'geographic': geo_support * self.weights['geographic']
        }
        
        # Combine with initial CNN prediction
        initial_conf = initial_prediction['confidence']
        evidence_boost = sum(evidence_scores.values())
        
        # Refined confidence (simple linear combination)
        refined_conf = min(0.99, initial_conf * 0.4 + evidence_boost * 0.6)
        
        return {
            'class': initial_prediction['class'],
            'confidence': refined_conf,
            'initial_confidence': initial_conf,
            'evidence_scores': evidence_scores,
            'evidence_summary': self._summarize_evidence(retrieved_evidence)
        }
    
    def _analyze_visual_evidence(self, visual_results: List, predicted_class: str) -> float:
        """Calculate visual evidence support."""
        if not visual_results:
            return 0.5
        
        matching = sum(1 for r in visual_results if r['diagnosis'] == predicted_class)
        return matching / len(visual_results)
    
    def _analyze_clinical_evidence(self, clinical_results: List, predicted_class: str) -> float:
        """Calculate clinical evidence support."""
        if not clinical_results:
            return 0.5
        
        matching = sum(1 for r in clinical_results if r['diagnosis'] == predicted_class)
        return matching / len(clinical_results)
    
    def _analyze_lab_evidence(self, lab_results: Dict, predicted_class: str) -> float:
        """Calculate lab evidence support."""
        if not lab_results or 'pathogen_distribution' not in lab_results:
            return 0.5
        
        # Check if pathogens match predicted diagnosis type
        pathogen_dist = lab_results['pathogen_distribution']
        if not pathogen_dist:
            return 0.5
        
        # Simple heuristic: fungal pathogens support Eumycetoma, bacterial support Actinomycetoma
        if predicted_class == 'Eumycetoma':
            return 0.8 if any('Madurella' in p['pathogen'] or 'Scedosporium' in p['pathogen'] 
                            for p in pathogen_dist) else 0.3
        else:
            return 0.8 if any('Nocardia' in p['pathogen'] or 'Actinomadura' in p['pathogen'] 
                            or 'Streptomyces' in p['pathogen'] for p in pathogen_dist) else 0.3
    
    def _analyze_geographic_evidence(self, geo_results: Dict, predicted_class: str) -> float:
        """Calculate geographic evidence support."""
        if not geo_results:
            return 0.5
        
        if predicted_class == 'Eumycetoma':
            return geo_results.get('eumy_prevalence', 50) / 100.0
        else:
            return geo_results.get('actino_prevalence', 50) / 100.0
    
    def _summarize_evidence(self, retrieved_evidence: Dict) -> Dict:
        """Summarize retrieved evidence."""
        return {
            'visual_cases': len(retrieved_evidence.get('visual', [])),
            'clinical_matches': len(retrieved_evidence.get('clinical', [])),
            'lab_confirmations': retrieved_evidence.get('lab', {}).get('total_confirmed', 0),
            'literature_refs': len(retrieved_evidence.get('literature', []))
        }
