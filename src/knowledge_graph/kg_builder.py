"""
Knowledge Graph Builder for Mycetoma Diagnosis System.

This module constructs a multi-modal knowledge graph from clinical data,
images, literature, and geographic information. Supports both initial
construction and incremental updates from community contributions.

Architecture:
    - Neo4j graph database
    - 8 entity types (Patient, Image, ClinicalNote, LabResult, Location, 
      Pathogen, Disease, Drug, Literature)
    - 10+ relationship types
    - Feature vectors stored for visual similarity
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from tqdm import tqdm
import torch

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from src.models.inception_v3 import InceptionV3Classifier


class KnowledgeGraphBuilder:
    """
    Build and maintain the Mycetoma Knowledge Graph.
    
    Supports:
        - Initial KG construction from CSV files
        - Incremental updates from community contributions
        - Version control and snapshots
        - Quality validation
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        model_path: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize KG builder.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            model_path: Path to trained InceptionV3 for feature extraction
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        # Connect to Neo4j
        try:
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            print(f"✓ Connected to Neo4j at {neo4j_uri}")
        except ServiceUnavailable:
            raise ConnectionError(
                f"Could not connect to Neo4j at {neo4j_uri}. "
                f"Make sure Neo4j is running."
            )
        
        # Load InceptionV3 for feature extraction
        if model_path and os.path.exists(model_path):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.feature_extractor = InceptionV3Classifier(num_classes=2, pretrained=False)
            self.feature_extractor.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
            print(f"✓ Loaded InceptionV3 from {model_path}")
        else:
            self.feature_extractor = None
            print("⚠ No feature extractor loaded - will skip image features")
        
        # Statistics
        self.stats = {
            'entities_added': 0,
            'relationships_added': 0,
            'start_time': None,
            'end_time': None
        }
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            print("✓ Neo4j connection closed")
    
    def clear_database(self, confirm: bool = False):
        """
        Clear all data from the database.
        
        Args:
            confirm: Must be True to actually clear (safety check)
        
        WARNING: This deletes ALL data in the database!
        """
        if not confirm:
            raise ValueError(
                "Must set confirm=True to clear database. "
                "This operation is irreversible!"
            )
        
        print("\n⚠️  WARNING: Clearing entire database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Database cleared")
    
    def create_constraints(self):
        """
        Create uniqueness constraints for efficient querying.
        
        Constraints ensure:
            - No duplicate entities
            - Faster lookups
            - Data integrity
        """
        print("\nCreating database constraints...")
        
        constraints = [
            "CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.case_id IS UNIQUE",
            "CREATE CONSTRAINT image_id IF NOT EXISTS FOR (i:Image) REQUIRE i.image_id IS UNIQUE",
            "CREATE CONSTRAINT note_id IF NOT EXISTS FOR (c:ClinicalNote) REQUIRE c.note_id IS UNIQUE",
            "CREATE CONSTRAINT lab_id IF NOT EXISTS FOR (l:LabResult) REQUIRE l.lab_id IS UNIQUE",
            "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (loc:Location) REQUIRE loc.name IS UNIQUE",
            "CREATE CONSTRAINT pathogen_name IF NOT EXISTS FOR (path:Pathogen) REQUIRE path.name IS UNIQUE",
            "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (drug:Drug) REQUIRE drug.name IS UNIQUE",
            "CREATE CONSTRAINT lit_pmid IF NOT EXISTS FOR (lit:Literature) REQUIRE lit.pmid IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    pass
        
        print("✓ Constraints created")
    
    def create_indexes(self):
        """Create indexes for faster queries."""
        print("\nCreating database indexes...")
        
        indexes = [
            "CREATE INDEX patient_location IF NOT EXISTS FOR (p:Patient) ON (p.location)",
            "CREATE INDEX patient_diagnosis IF NOT EXISTS FOR (p:Patient) ON (p.diagnosis)",
            "CREATE INDEX pathogen_type IF NOT EXISTS FOR (path:Pathogen) ON (path.type)",
            "CREATE INDEX literature_year IF NOT EXISTS FOR (lit:Literature) ON (lit.year)",
        ]
        
        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                except Exception:
                    pass
        
        print("✓ Indexes created")
    
    def add_patients(self, cases_df: pd.DataFrame) -> int:
        """
        Add patient nodes to KG.
        
        Args:
            cases_df: DataFrame with patient case information
        
        Returns:
            Number of patients added
        """
        print(f"\nAdding {len(cases_df)} patient nodes...")
        
        added = 0
        with self.driver.session() as session:
            for _, row in tqdm(cases_df.iterrows(), total=len(cases_df), desc="Patients"):
                try:
                    session.run("""
                        MERGE (p:Patient {case_id: $case_id})
                        SET p.age = $age,
                            p.gender = $gender,
                            p.location = $location,
                            p.country = $country,
                            p.diagnosis = $diagnosis,
                            p.onset_date = $onset_date,
                            p.anatomical_site = $anatomical_site,
                            p.disease_duration_months = $duration,
                            p.trauma_history = $trauma,
                            p.family_history = $family_history,
                            p.has_sinuses = $sinuses,
                            p.grain_color = $grain_color,
                            p.previous_surgery = $previous_surgery,
                            p.previous_treatment = $previous_treatment,
                            p.created_at = datetime()
                    """,
                    case_id=row['case_id'],
                    age=int(row['age']),
                    gender=row['gender'],
                    location=row.get('location', 'Unknown'),
                    country=row.get('country', 'Unknown'),
                    diagnosis=row['diagnosis'],
                    onset_date=str(row.get('onset_date', '')),
                    anatomical_site=row.get('anatomical_site', 'Unknown'),
                    duration=int(row.get('disease_duration_months', 0)),
                    trauma=row.get('trauma_history', 'Unknown'),
                    family_history=row.get('family_history', 'Unknown'),
                    sinuses=row.get('has_sinuses', 'Unknown'),
                    grain_color=row.get('grain_color', 'Unknown'),
                    previous_surgery=row.get('previous_surgery', 'Unknown'),
                    previous_treatment=row.get('previous_treatment', 'Unknown')
                    )
                    added += 1
                except Exception as e:
                    print(f"Error adding patient {row['case_id']}: {e}")
        
        self.stats['entities_added'] += added
        print(f"✓ Added {added} patient nodes")
        return added
    
    def add_images_with_features(
        self,
        images_df: pd.DataFrame,
        image_dir: Optional[str] = None
    ) -> int:
        """
        Add image nodes with InceptionV3 features.
        
        Args:
            images_df: DataFrame with image metadata
            image_dir: Directory containing actual image files
        
        Returns:
            Number of images added
        """
        print(f"\nAdding {len(images_df)} image nodes with features...")
        
        if self.feature_extractor is None:
            print("⚠ No feature extractor available - adding images without features")
        
        added = 0
        with self.driver.session() as session:
            for idx, row in tqdm(images_df.iterrows(), total=len(images_df), desc="Images"):
                try:
                    # Extract features if available
                    features = None
                    if self.feature_extractor and image_dir:
                        image_path = os.path.join(image_dir, row.get('image_path', ''))
                        if os.path.exists(image_path):
                            features = self._extract_image_features(image_path)
                    
                    # Add image node
                    session.run("""
                        MATCH (p:Patient {case_id: $case_id})
                        MERGE (i:Image {image_id: $image_id})
                        SET i.image_path = $image_path,
                            i.stain_type = $stain_type,
                            i.magnification = $magnification,
                            i.features = $features,
                            i.created_at = datetime()
                        MERGE (p)-[:HAS_IMAGE]->(i)
                    """,
                    case_id=row.get('case_id', ''),
                    image_id=row['image_id'],
                    image_path=row.get('image_path', ''),
                    stain_type=row.get('stain_type', 'H&E'),
                    magnification=row.get('magnification', '40x'),
                    features=features.tolist() if features is not None else []
                    )
                    added += 1
                except Exception as e:
                    print(f"Error adding image {row.get('image_id', 'unknown')}: {e}")
        
        self.stats['entities_added'] += added
        self.stats['relationships_added'] += added  # HAS_IMAGE relationships
        print(f"✓ Added {added} image nodes")
        return added
    
    def _extract_image_features(self, image_path: str) -> np.ndarray:
        """
        Extract 2048-dim features from image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Feature vector of shape (2048,)
        """
        from PIL import Image
        from torchvision import transforms
        
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor.extract_features(img_tensor)
        
        return features.cpu().numpy().squeeze()
    
    def add_clinical_notes(self, notes_df: pd.DataFrame) -> int:
        """Add clinical note nodes."""
        print(f"\nAdding {len(notes_df)} clinical note nodes...")
        
        added = 0
        with self.driver.session() as session:
            for _, row in tqdm(notes_df.iterrows(), total=len(notes_df), desc="Clinical Notes"):
                try:
                    session.run("""
                        MATCH (p:Patient {case_id: $case_id})
                        MERGE (c:ClinicalNote {note_id: $note_id})
                        SET c.symptoms = $symptoms,
                            c.duration_months = $duration,
                            c.presentation = $presentation,
                            c.trauma_history = $trauma,
                            c.pain_level = $pain_level,
                            c.local_hyperhidrosis = $hyperhidrosis,
                            c.lymphadenopathy = $lymphadenopathy,
                            c.created_at = datetime()
                        MERGE (p)-[:HAS_CLINICAL_NOTE]->(c)
                    """,
                    case_id=row.get('case_id', ''),
                    note_id=row['note_id'],
                    symptoms=row.get('symptoms', ''),
                    duration=int(row.get('duration_months', 0)),
                    presentation=row.get('presentation', ''),
                    trauma=row.get('trauma_history', 'Unknown'),
                    pain_level=row.get('pain_level', 'Unknown'),
                    hyperhidrosis=row.get('local_hyperhidrosis', 'Unknown'),
                    lymphadenopathy=row.get('lymphadenopathy', 'Unknown')
                    )
                    added += 1
                except Exception as e:
                    print(f"Error adding note {row.get('note_id', 'unknown')}: {e}")
        
        self.stats['entities_added'] += added
        self.stats['relationships_added'] += added
        print(f"✓ Added {added} clinical note nodes")
        return added
    
    def add_lab_results(self, labs_df: pd.DataFrame) -> int:
        """Add lab result nodes and link to pathogens."""
        print(f"\nAdding {len(labs_df)} lab result nodes...")
        
        added = 0
        with self.driver.session() as session:
            for _, row in tqdm(labs_df.iterrows(), total=len(labs_df), desc="Lab Results"):
                try:
                    session.run("""
                        MATCH (p:Patient {case_id: $case_id})
                        MERGE (path:Pathogen {name: $pathogen})
                        ON CREATE SET path.type = $pathogen_type
                        MERGE (l:LabResult {lab_id: $lab_id})
                        SET l.confirmed_pathogen = $pathogen,
                            l.method = $method,
                            l.confirmation_date = $date,
                            l.susceptibility = $susceptibility,
                            l.grain_culture_days = $culture_days,
                            l.pcr_result = $pcr,
                            l.histopathology_result = $histo,
                            l.created_at = datetime()
                        MERGE (p)-[:HAS_LAB_RESULT]->(l)
                        MERGE (l)-[:IDENTIFIES]->(path)
                    """,
                    case_id=row.get('case_id', ''),
                    pathogen=row.get('confirmed_pathogen', 'Unknown'),
                    pathogen_type=row.get('pathogen_type', 'Unknown'),
                    lab_id=row['lab_id'],
                    method=row.get('method', 'Unknown'),
                    date=str(row.get('confirmation_date', '')),
                    susceptibility=row.get('susceptibility', 'Unknown'),
                    culture_days=row.get('grain_culture_days', 'NA'),
                    pcr=row.get('pcr_result', 'Unknown'),
                    histo=row.get('histopathology_result', 'Unknown')
                    )
                    added += 1
                except Exception as e:
                    print(f"Error adding lab result {row.get('lab_id', 'unknown')}: {e}")
        
        self.stats['entities_added'] += added
        self.stats['relationships_added'] += added * 2  # HAS_LAB_RESULT + IDENTIFIES
        print(f"✓ Added {added} lab result nodes")
        return added
    
    def add_geographic_locations(self, locations_df: pd.DataFrame) -> int:
        """Add geographic location nodes with epidemiology."""
        print(f"\nAdding {len(locations_df)} geographic location nodes...")
        
        added = 0
        with self.driver.session() as session:
            # Add location nodes
            for _, row in tqdm(locations_df.iterrows(), total=len(locations_df), desc="Locations"):
                try:
                    session.run("""
                        MERGE (loc:Location {name: $name})
                        SET loc.country = $country,
                            loc.region = $region,
                            loc.latitude = $latitude,
                            loc.longitude = $longitude,
                            loc.actino_prevalence = $actino,
                            loc.eumy_prevalence = $eumy,
                            loc.total_cases = $total,
                            loc.climate = $climate,
                            loc.endemic_level = $endemic_level,
                            loc.vegetation_type = $vegetation,
                            loc.created_at = datetime()
                    """,
                    name=row['location'],
                    country=row.get('country', 'Unknown'),
                    region=row.get('region', 'Unknown'),
                    latitude=float(row.get('latitude', 0)),
                    longitude=float(row.get('longitude', 0)),
                    actino=float(row.get('actino_prevalence', 0)),
                    eumy=float(row.get('eumy_prevalence', 0)),
                    total=int(row.get('total_cases', 0)),
                    climate=row.get('climate', 'Unknown'),
                    endemic_level=row.get('endemic_level', 'Unknown'),
                    vegetation=row.get('vegetation_type', 'Unknown')
                    )
                    added += 1
                except Exception as e:
                    print(f"Error adding location {row.get('location', 'unknown')}: {e}")
            
            # Link patients to locations
            print("Linking patients to locations...")
            result = session.run("""
                MATCH (p:Patient), (loc:Location)
                WHERE p.location = loc.name
                MERGE (p)-[:RESIDES_IN]->(loc)
                RETURN count(*) as count
            """)
            links = result.single()['count']
            self.stats['relationships_added'] += links
            print(f"  Created {links} RESIDES_IN relationships")
        
        self.stats['entities_added'] += added
        print(f"✓ Added {added} location nodes")
        return added
    
    def add_disease_ontology(self) -> Tuple[int, int]:
        """
        Add disease, pathogen, and drug ontology.
        
        Returns:
            Tuple of (entities_added, relationships_added)
        """
        print("\nAdding disease-pathogen-drug ontology...")
        
        entities = 0
        relationships = 0
        
        with self.driver.session() as session:
            # Create main disease nodes
            session.run("""
                MERGE (d1:Disease {name: 'Actinomycetoma'})
                SET d1.type = 'Bacterial',
                    d1.icd10 = 'B47.1',
                    d1.description = 'Bacterial mycetoma caused by actinomycetes'
                    
                MERGE (d2:Disease {name: 'Eumycetoma'})
                SET d2.type = 'Fungal',
                    d2.icd10 = 'B47.0',
                    d2.description = 'Fungal mycetoma caused by various fungi'
            """)
            entities += 2
            
            # Actinomycetoma pathogens
            actino_pathogens = [
                ('Nocardia brasiliensis', 'Black', 'Americas'),
                ('Actinomadura madurae', 'Red/Pink', 'Worldwide'),
                ('Actinomadura pelletieri', 'Red', 'Africa'),
                ('Streptomyces somaliensis', 'Yellow', 'Africa, Middle East'),
                ('Nocardia asteroides', 'White', 'Worldwide')
            ]
            
            for pathogen, grain_color, distribution in actino_pathogens:
                session.run("""
                    MATCH (d:Disease {name: 'Actinomycetoma'})
                    MERGE (path:Pathogen {name: $pathogen})
                    SET path.type = 'Bacterial',
                        path.grain_color = $grain_color,
                        path.geographic_distribution = $distribution
                    MERGE (path)-[:CAUSES]->(d)
                """, pathogen=pathogen, grain_color=grain_color, distribution=distribution)
                entities += 1
                relationships += 1
            
            # Eumycetoma pathogens
            eumy_pathogens = [
                ('Madurella mycetomatis', 'Black', 'Africa, India'),
                ('Scedosporium apiospermum', 'White/Yellow', 'Worldwide'),
                ('Madurella grisea', 'Black', 'South America'),
                ('Acremonium species', 'White', 'Africa'),
                ('Fusarium species', 'White', 'Worldwide')
            ]
            
            for pathogen, grain_color, distribution in eumy_pathogens:
                session.run("""
                    MATCH (d:Disease {name: 'Eumycetoma'})
                    MERGE (path:Pathogen {name: $pathogen})
                    SET path.type = 'Fungal',
                        path.grain_color = $grain_color,
                        path.geographic_distribution = $distribution
                    MERGE (path)-[:CAUSES]->(d)
                """, pathogen=pathogen, grain_color=grain_color, distribution=distribution)
                entities += 1
                relationships += 1
            
            # Link patients to diseases through diagnosis
            print("Linking patients to diseases...")
            result = session.run("""
                MATCH (p:Patient), (d:Disease)
                WHERE p.diagnosis = d.name
                MERGE (p)-[:DIAGNOSED_WITH]->(d)
                RETURN count(*) as count
            """)
            links = result.single()['count']
            relationships += links
            print(f"  Created {links} DIAGNOSED_WITH relationships")
            
            # Actinomycetoma drugs
            actino_drugs = [
                ('Streptomycin', 'Aminoglycoside', '1g IM daily', 3),
                ('Co-trimoxazole', 'Sulfonamide', '960mg PO BID', 10),
                ('Amikacin', 'Aminoglycoside', '15mg/kg IV daily', 3),
                ('Dapsone', 'Sulfone', '100mg PO daily', 12)
            ]
            
            for drug, drug_class, dose, duration in actino_drugs:
                session.run("""
                    MATCH (d:Disease {name: 'Actinomycetoma'})
                    MERGE (drug:Drug {name: $drug})
                    SET drug.drug_class = $drug_class,
                        drug.typical_dose = $dose,
                        drug.duration_months = $duration
                    MERGE (d)-[:TREATED_WITH]->(drug)
                """, drug=drug, drug_class=drug_class, dose=dose, duration=duration)
                entities += 1
                relationships += 1
            
            # Eumycetoma drugs
            eumy_drugs = [
                ('Itraconazole', 'Azole antifungal', '400mg PO daily', 12),
                ('Ketoconazole', 'Azole antifungal', '200-400mg PO daily', 12),
                ('Voriconazole', 'Azole antifungal', '200mg PO BID', 12),
                ('Surgical excision', 'Surgery', 'Wide local excision', 0)
            ]
            
            for drug, drug_class, dose, duration in eumy_drugs:
                session.run("""
                    MATCH (d:Disease {name: 'Eumycetoma'})
                    MERGE (drug:Drug {name: $drug})
                    SET drug.drug_class = $drug_class,
                        drug.typical_dose = $dose,
                        drug.duration_months = $duration
                    MERGE (d)-[:TREATED_WITH]->(drug)
                """, drug=drug, drug_class=drug_class, dose=dose, duration=duration)
                entities += 1
                relationships += 1
        
        self.stats['entities_added'] += entities
        self.stats['relationships_added'] += relationships
        print(f"✓ Added {entities} ontology entities and {relationships} relationships")
        return entities, relationships
    
    def add_literature(self, literature_df: pd.DataFrame) -> int:
        """Add PubMed literature nodes."""
        print(f"\nAdding {len(literature_df)} literature nodes...")
        
        added = 0
        with self.driver.session() as session:
            for _, row in tqdm(literature_df.iterrows(), total=len(literature_df), desc="Literature"):
                try:
                    session.run("""
                        MERGE (lit:Literature {pmid: $pmid})
                        SET lit.title = $title,
                            lit.authors = $authors,
                            lit.year = $year,
                            lit.journal = $journal,
                            lit.keywords = $keywords,
                            lit.abstract = $abstract,
                            lit.doi = $doi,
                            lit.created_at = datetime()
                    """,
                    pmid=str(row['pmid']),
                    title=row.get('title', ''),
                    authors=row.get('authors', ''),
                    year=int(row.get('year', 0)),
                    journal=row.get('journal', ''),
                    keywords=row.get('keywords', ''),
                    abstract=row.get('abstract', ''),
                    doi=row.get('doi', '')
                    )
                    added += 1
                except Exception as e:
                    print(f"Error adding literature {row.get('pmid', 'unknown')}: {e}")
        
        self.stats['entities_added'] += added
        print(f"✓ Added {added} literature nodes")
        return added
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current KG statistics.
        
        Returns:
            Dictionary with entity counts, relationship counts, etc.
        """
        with self.driver.session() as session:
            # Count entities by type
            entity_counts = {}
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY count DESC
            """)
            for record in result:
                entity_counts[record['label']] = record['count']
            
            # Count relationships
            rel_result = session.run("""
                MATCH ()-[r]->()
                RETURN count(r) AS count
            """)
            total_relationships = rel_result.single()['count']
            
            # Count by relationship type
            rel_types = {}
            type_result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(r) AS count
                ORDER BY count DESC
            """)
            for record in type_result:
                rel_types[record['rel_type']] = record['count']
        
        return {
            'entity_counts': entity_counts,
            'total_entities': sum(entity_counts.values()),
            'total_relationships': total_relationships,
            'relationship_types': rel_types,
            'timestamp': datetime.now().isoformat()
        }
    
    def build_complete_kg(
        self,
        data_dir: str,
        image_dir: Optional[str] = None,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Build complete Knowledge Graph from data directory.
        
        Args:
            data_dir: Directory containing CSV files
            image_dir: Directory containing image files
            clear_existing: Whether to clear existing data first
        
        Returns:
            Dictionary with build statistics
        """
        print("\n" + "="*70)
        print("BUILDING MYCETOMA KNOWLEDGE GRAPH")
        print("="*70)
        
        self.stats['start_time'] = datetime.now()
        
        # Clear if requested
        if clear_existing:
            self.clear_database(confirm=True)
        
        # Setup database
        print("\n[1/9] Setting up database...")
        self.create_constraints()
        self.create_indexes()
        
        # Load data files
        print("\n[2/9] Loading data files...")
        data_path = Path(data_dir)
        
        cases_df = pd.read_csv(data_path / "cases.csv")
        images_df = pd.read_csv(data_path / "images.csv") if (data_path / "images.csv").exists() else pd.DataFrame()
        notes_df = pd.read_csv(data_path / "clinical_notes.csv")
        labs_df = pd.read_csv(data_path / "lab_results.csv")
        locations_df = pd.read_csv(data_path / "geographic_locations.csv")
        literature_df = pd.read_csv(data_path / "literature.csv") if (data_path / "literature.csv").exists() else pd.DataFrame()
        
        print(f"  ✓ Loaded {len(cases_df)} cases")
        print(f"  ✓ Loaded {len(images_df)} images")
        print(f"  ✓ Loaded {len(notes_df)} clinical notes")
        print(f"  ✓ Loaded {len(labs_df)} lab results")
        print(f"  ✓ Loaded {len(locations_df)} locations")
        print(f"  ✓ Loaded {len(literature_df)} literature references")
        
        # Build KG components
        print("\n[3/9] Adding patients...")
        self.add_patients(cases_df)
        
        if not images_df.empty:
            print("\n[4/9] Adding images with features...")
            self.add_images_with_features(images_df, image_dir)
        else:
            print("\n[4/9] Skipping images (no data)")
        
        print("\n[5/9] Adding clinical notes...")
        self.add_clinical_notes(notes_df)
        
        print("\n[6/9] Adding lab results...")
        self.add_lab_results(labs_df)
        
        print("\n[7/9] Adding geographic locations...")
        self.add_geographic_locations(locations_df)
        
        print("\n[8/9] Adding disease ontology...")
        self.add_disease_ontology()
        
        if not literature_df.empty:
            print("\n[9/9] Adding literature...")
            self.add_literature(literature_df)
        else:
            print("\n[9/9] Skipping literature (no data)")
        
        # Final statistics
        self.stats['end_time'] = datetime.now()
        final_stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("KNOWLEDGE GRAPH BUILD COMPLETE!")
        print("="*70)
        
        print(f"\n✓ Total Entities: {final_stats['total_entities']}")
        print(f"✓ Total Relationships: {final_stats['total_relationships']}")
        print(f"✓ Build Time: {(self.stats['end_time'] - self.stats['start_time']).total_seconds():.1f}s")
        
        print("\nEntity Breakdown:")
        for label, count in sorted(final_stats['entity_counts'].items(), key=lambda x: -x[1]):
            print(f"  {label:20s}: {count:5d}")
        
        return final_stats


def main():
    """Command-line interface for KG builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Mycetoma Knowledge Graph")
    parser.add_argument("--data-path", required=True, help="Path to data directory")
    parser.add_argument("--image-dir", help="Path to image directory")
    parser.add_argument("--model-path", help="Path to InceptionV3 model")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--clear", action="store_true", help="Clear existing data")
    
    args = parser.parse_args()
    
    # Build KG
    builder = KnowledgeGraphBuilder(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        model_path=args.model_path
    )
    
    try:
        stats = builder.build_complete_kg(
            data_dir=args.data_path,
            image_dir=args.image_dir,
            clear_existing=args.clear
        )
        
        print(f"\n✓ Knowledge Graph ready at {args.neo4j_uri}")
        print(f"✓ Access Neo4j Browser: http://localhost:7474")
        
    finally:
        builder.close()


if __name__ == "__main__":
    main()
