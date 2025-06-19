import json
import os
from typing import List, Dict, Any
import logging

from ..models.schemas import CoTExample, ReasoningPattern

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Get the absolute path to the data directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to backend, then up to project root, then to data
            self.data_dir = os.path.join(current_dir, "..", "..", "..", "data")
        else:
            self.data_dir = data_dir
        
    def list_available_datasets(self) -> List[str]:
        """List all available datasets"""
        try:
            datasets = []
            if os.path.exists(self.data_dir):
                for filename in os.listdir(self.data_dir):
                    if filename.endswith('.js'):
                        dataset_name = filename.replace('.js', '').replace('-', '_')
                        datasets.append(dataset_name)
            return sorted(datasets)
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return []
    
    def load_dataset(self, dataset_name: str) -> List[CoTExample]:
        """Load a specific dataset and convert to CoTExample objects"""
        try:
            # Map dataset names to files (prefer JSON over JS)
            dataset_files = {
                'pure_logic_cots': 'pure-logic-cots.js',
                'mixed_cots': 'mixed-cots.js',
                'abstract_cots': 'abstract-cots.js',
                'cots': 'cots.json'  # Use JSON version for easier parsing
            }
            
            if dataset_name not in dataset_files:
                raise FileNotFoundError(f"Dataset '{dataset_name}' not found")
            
            file_path = os.path.join(self.data_dir, dataset_files[dataset_name])
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file '{file_path}' not found")
            
            # Check if it's a JSON file or JavaScript file
            if file_path.endswith('.json'):
                # Read JSON file directly
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            else:
                # Read JavaScript export file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract JSON data from JavaScript export
                # Find the array content between [ and ]
                start_idx = content.find('[')
                end_idx = content.rfind('];')
                
                if start_idx == -1 or end_idx == -1:
                    raise ValueError(f"Could not parse dataset file: {file_path}")
                
                json_str = content[start_idx:end_idx+1]
                
                # Convert JavaScript object notation to JSON
                # Replace single quotes with double quotes
                json_str = json_str.replace("'", '"')
                
                # Handle template literals and multi-line strings
                json_str = self._clean_js_to_json(json_str)
                
                # Parse JSON
                try:
                    raw_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    # Fallback: try to manually parse the JavaScript array
                    raw_data = self._parse_js_array_manually(content)
            
            # Convert to CoTExample objects
            cot_examples = []
            for i, item in enumerate(raw_data):
                try:
                    # Generate ID if not present
                    cot_id = item.get('id', f"{dataset_name}_{i}")
                    
                    # Map reasoning pattern
                    reasoning_pattern = None
                    if 'reasoning_pattern' in item:
                        pattern_str = item['reasoning_pattern'].lower().replace(' ', '_')
                        try:
                            reasoning_pattern = ReasoningPattern(pattern_str)
                        except ValueError:
                            logger.warning(f"Unknown reasoning pattern: {item['reasoning_pattern']}")
                    
                    cot_example = CoTExample(
                        id=cot_id,
                        question=item.get('question', ''),
                        answer=item.get('answer', ''),
                        cot=item.get('cot', ''),
                        reasoning_pattern=reasoning_pattern
                    )
                    
                    cot_examples.append(cot_example)
                    
                except Exception as e:
                    logger.warning(f"Error processing item {i} in dataset {dataset_name}: {e}")
                    continue
            
            logger.info(f"Loaded {len(cot_examples)} CoT examples from {dataset_name}")
            return cot_examples
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            raise
    
    def _clean_js_to_json(self, js_str: str) -> str:
        """Clean JavaScript object notation to valid JSON"""
        # This is a simplified converter - in production you'd want a proper JS parser
        
        # Replace unquoted keys with quoted keys
        import re
        
        # Match unquoted object keys
        js_str = re.sub(r'(\w+):', r'"\1":', js_str)
        
        # Fix any double-quoted quotes
        js_str = js_str.replace('""', '"')
        
        # Remove trailing commas
        js_str = re.sub(r',(\s*[}\]])', r'\1', js_str)
        
        return js_str
    
    def _parse_js_array_manually(self, content: str) -> List[Dict[str, Any]]:
        """Manually parse JavaScript array when JSON parsing fails"""
        # This is a simple fallback - extract objects between { and }
        import re
        
        # Find all objects in the array
        objects = []
        
        # Look for patterns like { id: 'cot-1', question: '...', ... }
        object_pattern = r'\{\s*id:\s*[\'"]([^\'"]+)[\'"],\s*question:\s*[\'"]([^\'"]*)[\'"],\s*answer:\s*[\'"]([^\'"]*)[\'"],\s*cot:\s*[\'"]([^\'"]*)[\'"]'
        
        matches = re.finditer(object_pattern, content, re.DOTALL)
        
        for match in matches:
            obj = {
                'id': match.group(1),
                'question': match.group(2),
                'answer': match.group(3),
                'cot': match.group(4)
            }
            objects.append(obj)
        
        logger.info(f"Manually parsed {len(objects)} objects from JavaScript file")
        return objects
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save experiment results to JSON file"""
        try:
            output_path = os.path.join(self.data_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results to {filename}: {e}")
            raise
    
    def load_ground_truth(self, dataset_name: str) -> Dict[str, str]:
        """Load ground truth labels for evaluation"""
        # This would typically come from a separate annotation file
        # For now, return the known ground truth for pure-logic dataset
        
        if dataset_name == 'pure_logic_cots':
            return {
                "What should I check if my car won't start in cold weather?|||Check the battery, engine oil viscosity, and ensure the fuel system is working properly.": "correct",
                "How do I know if my car battery needs replacement?|||Signs include slow engine cranking, dim headlights, dashboard warning lights, or the battery is over 3-4 years old.": "correct",
                "What's the difference between 0W-20 and 5W-30 motor oil?|||0W-20 flows better in cold weather and provides better fuel economy, while 5W-30 offers better protection at higher temperatures.": "correct",
                "What should I do if my car is overheating?|||Pull over safely, turn off the engine, wait for it to cool down, check coolant levels, and call for assistance if needed.": "incorrect",
                "How do I properly jump-start a car?|||Connect positive cable to dead battery positive terminal, other end to good battery positive, then negative cable to good battery negative and final connection to ground point on dead car.": "incorrect"
            }
        
        return {}
    
    def export_for_research(self, data: Dict[str, Any], filename: str = "research_export.json") -> None:
        """Export data in format suitable for research papers"""
        try:
            # Create research-friendly format
            research_data = {
                "metadata": {
                    "experiment_name": "reasoning_pattern_clustering",
                    "description": "CoT reasoning pattern clustering for hallucination detection",
                    "total_cots": len(data.get('cot_examples', [])),
                    "total_qa_pairs": len(data.get('qa_pairs', {})),
                    "clusters_found": data.get('clustering_summary', {}).get('clusters', 0),
                    "human_effort_percentage": None,  # To be calculated
                    "automation_rate": None  # To be calculated
                },
                "results": data
            }
            
            # Calculate key metrics
            qa_pairs = data.get('qa_pairs', {})
            if qa_pairs:
                total_pairs = len(qa_pairs)
                human_labeled = sum(1 for qa in qa_pairs.values() if qa.get('source') == 'HUMAN')
                propagated = sum(1 for qa in qa_pairs.values() if qa.get('source') == 'PROPAGATED')
                
                research_data["metadata"]["human_effort_percentage"] = (human_labeled / total_pairs) * 100 if total_pairs > 0 else 0
                research_data["metadata"]["automation_rate"] = (propagated / total_pairs) * 100 if total_pairs > 0 else 0
            
            self.save_results(research_data, filename)
            
        except Exception as e:
            logger.error(f"Error exporting research data: {e}")
            raise 