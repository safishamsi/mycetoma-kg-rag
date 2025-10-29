"""
Multi-modal retrieval from Knowledge Graph.

Retrieves evidence across 5 modalities:
1. Visual similarity (image features)
2. Clinical text matching (symptoms)
3. Laboratory confirmations (pathogens)
4. Geographic epidemiology (location priors)
5. Literature references (PubMed)
"""

import numpy as np
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase


class MultiModalRetriever:
    """Retrieve evidence from KG across multiple modalities."""
    
    def __init__(self, kg_driver, config: Dict = None):
        self.driver = kg_driver
        self.config = config or {}
        
        # Weights from paper
        self.weights = {
            'visual': self.config.get('retrieval', {}).get('visual', {}).get('weight', 0.35),
            'clinical': self.config.get('retrieval', {}).get('clinical', {}).get('weight', 0.20),
            'lab': self.config.get('retrieval', {}).get('lab', {}).get('weight', 0.30),
            'geographic': self.config.get('retrieval', {}).get('geographic', {}).get('weight', 0.10),
            'literature': self.config.get('retrieval', {}).get('literature', {}).get('weight', 0.05)
        }
    
    def retrieve_visual_similar(self, query_features: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Retrieve top-k visually similar cases.
        
        Args:
            query_features: 2048-dim feature vector
            k: Number of results
        
        Returns:
            List of similar cases with similarity scores
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Image)
                WHERE size(i.features) > 0
                WITH i,
                     reduce(dot = 0.0, idx IN range(0, size(i.features)-1) | 
                         dot + i.features[idx] * $query_features[idx]
                     ) AS dotProduct,
                     sqrt(reduce(sum = 0.0, x IN i.features | sum + x*x)) AS norm1,
                     sqrt(reduce(sum = 0.0, x IN $query_features | sum + x*x)) AS norm2
                WITH i, dotProduct / (norm1 * norm2) AS similarity
                WHERE similarity > 0.5
                ORDER BY similarity DESC
                LIMIT $k
                MATCH (p:Patient)-[:HAS_IMAGE]->(i)
                RETURN p.case_id AS case_id,
                       p.diagnosis AS diagnosis,
                       similarity,
                       i.image_id AS image_id
            """, query_features=query_features.tolist(), k=k)
            
            return [dict(record) for record in result]
    
    def retrieve_clinical_matches(self, symptoms: str, k: int = 10) -> List[Dict]:
        """Retrieve cases with similar clinical presentations."""
        # Simple keyword matching (can be upgraded to semantic search)
        keywords = symptoms.lower().split()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Patient)-[:HAS_CLINICAL_NOTE]->(c:ClinicalNote)
                WHERE any(keyword IN $keywords WHERE toLower(c.symptoms) CONTAINS keyword
                       OR toLower(c.presentation) CONTAINS keyword)
                WITH p, c, 
                     size([keyword IN $keywords WHERE toLower(c.symptoms) CONTAINS keyword]) AS matches
                ORDER BY matches DESC
                LIMIT $k
                RETURN p.case_id AS case_id,
                       p.diagnosis AS diagnosis,
                       c.symptoms AS symptoms,
                       matches AS match_score
            """, keywords=keywords, k=k)
            
            return [dict(record) for record in result]
    
    def retrieve_lab_confirmations(self, predicted_diagnosis: str) -> Dict:
        """Get lab confirmation statistics for diagnosis."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Patient {diagnosis: $diagnosis})-[:HAS_LAB_RESULT]->(l:LabResult)-[:IDENTIFIES]->(path:Pathogen)
                WITH path.name AS pathogen, count(*) AS count
                ORDER BY count DESC
                RETURN collect({pathogen: pathogen, count: count}) AS pathogen_distribution,
                       sum(count) AS total_confirmed
            """, diagnosis=predicted_diagnosis)
            
            record = result.single()
            return dict(record) if record else {}
    
    def retrieve_geographic_priors(self, location: str) -> Dict:
        """Get geographic epidemiology for location."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (loc:Location {name: $location})
                RETURN loc.actino_prevalence AS actino_prevalence,
                       loc.eumy_prevalence AS eumy_prevalence,
                       loc.total_cases AS total_cases,
                       loc.climate AS climate
            """, location=location)
            
            record = result.single()
            return dict(record) if record else {}
    
    def retrieve_literature(self, diagnosis: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant literature."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (lit:Literature)
                WHERE toLower(lit.title) CONTAINS toLower($diagnosis)
                   OR toLower(lit.keywords) CONTAINS toLower($diagnosis)
                   OR toLower(lit.abstract) CONTAINS toLower($diagnosis)
                ORDER BY lit.year DESC
                LIMIT $k
                RETURN lit.pmid AS pmid,
                       lit.title AS title,
                       lit.year AS year,
                       lit.authors AS authors
            """, diagnosis=diagnosis, k=k)
            
            return [dict(record) for record in result]
    
    def retrieve_all_modalities(
        self,
        query_image_features: np.ndarray,
        query_symptoms: str,
        query_demographics: Dict,
        predicted_class: str,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve evidence from all modalities.
        
        Returns:
            Dictionary with results from each modality
        """
        results = {
            'visual': self.retrieve_visual_similar(query_image_features, k),
            'clinical': self.retrieve_clinical_matches(query_symptoms, k),
            'lab': self.retrieve_lab_confirmations(predicted_class),
            'geographic': self.retrieve_geographic_priors(query_demographics.get('location', 'Unknown')),
            'literature': self.retrieve_literature(predicted_class, 5)
        }
        
        return results
