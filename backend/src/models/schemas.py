from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class ReasoningPattern(str, Enum):
    DEDUCTIVE = "deductive"
    EXPERIENCE_BASED = "experience_based"
    SYSTEMS_THINKING = "systems_thinking"
    PROCEDURAL = "procedural"
    ANALOGICAL = "analogical"

class CoTExample(BaseModel):
    id: str
    question: str
    answer: str
    cot: str
    cluster_id: Optional[int] = None
    outlier_score: Optional[float] = None
    reasoning_pattern: Optional[ReasoningPattern] = None
    embedding: Optional[List[float]] = None

class QAPair(BaseModel):
    question: str
    answer: str
    cots: List[CoTExample]
    clusters: List[int]
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    propagation_source: Optional[str] = None
    shared_reasoning: Optional[str] = None

class ClusteringRequest(BaseModel):
    min_cluster_size: int = 2
    min_samples: int = 1

class ClusteringResponse(BaseModel):
    success: bool
    data: List[CoTExample]
    summary: Dict[str, Any]

class LabelRequest(BaseModel):
    qa_pair_key: str
    label: str  # 'correct' or 'incorrect'

class PropagationRequest(BaseModel):
    human_labels: Dict[str, str]
    num_representatives: int = 2

class PropagationResponse(BaseModel):
    success: bool
    qa_pairs: List[QAPair]
    summary: Dict[str, Any]
    evaluation: Optional[Dict[str, float]] = None

class ExperimentResults(BaseModel):
    cot_examples: List[CoTExample]
    qa_pairs: Dict[str, QAPair]
    clustering_summary: Dict[str, Any]
    propagation_summary: Dict[str, Any]
    evaluation_metrics: Dict[str, float] 