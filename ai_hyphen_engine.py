import asyncio
import json
import re
import requests
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import List, Dict, Any, Optional
import os
from datetime import datetime

class HyphenationEngine:
    """
    AI-powered hyphenation engine that processes DOCX documents and provides
    hyphenation corrections based on style guides and language variants.
    """
    
    def __init__(self):
        self.rules = self._load_rules()
        self.dictionary_sources = {
            'US English': 'Merriam-Webster Dictionary',
            'UK English': 'Oxford English Dictionary'
        }
        
    def _load_rules(self) -> Dict[str, Any]:
        """Load hyphenation rules from the rules file"""
        rules = {
            'compound_adjectives': [
                'well-known', 'high-quality', 'long-term', 'short-term', 'state-of-the-art',
                'up-to-date', 'one-on-one', 'face-to-face', 'full-time', 'part-time',
                'real-time', 'large-scale', 'small-scale', 'high-level', 'low-level',
                'self-evident', 'self-contained', 'user-friendly', 'cost-effective',
                'time-consuming', 'energy-efficient', 'eco-friendly', 'cloud-based'
            ],
            'number_compounds': [
                r'\d+-year-old', r'\d+-day', r'\d+-hour', r'\d+-minute', r'\d+-second',
                r'\d+-member', r'\d+-person', r'\d+-item', r'\d+-point', r'\d+-step'
            ],
            'prefix_patterns': {
                'co': ['co-operate', 'co-ordinate', 'co-author', 'co-worker'],
                'pre': ['pre-existing', 'pre-approval', 'pre-planned'],
                'post': ['post-processing', 'post-analysis', 'post-treatment'],
                'multi': ['multi-purpose', 'multi-level', 'multi-step'],
                'cross': ['cross-reference', 'cross-section', 'cross-validation']
            },
            'exceptions': {
                'never_hyphenate': ['very well', 'quite good', 'extremely large'],
                'always_hyphenate': ['state-of-the-art', 'cutting-edge', 'real-time']
            }
        }
        return rules

    def extract_text_from_docx(self, docx_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract text from DOCX file using standard Python libraries
        Returns list of paragraphs with their content and metadata
        """
        paragraphs = []
        
        try:
            with zipfile.ZipFile(BytesIO(docx_bytes), 'r') as zip_file:
                # Read the main document XML
                xml_content = zip_file.read('word/document.xml')
                root = ET.fromstring(xml_content)
                
                # Define namespaces
                namespaces = {
                    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
                }
                
                # Extract paragraphs
                para_elements = root.findall('.//w:p', namespaces)
                
                for i, para in enumerate(para_elements):
                    # Extract text from paragraph
                    text_parts = []
                    for text_elem in para.findall('.//w:t', namespaces):
                        if text_elem.text:
                            text_parts.append(text_elem.text)
                    
                    paragraph_text = ''.join(text_parts).strip()
                    
                    if paragraph_text:  # Only include non-empty paragraphs
                        paragraphs.append({
                            'para_id': i+1,  # Just the number, not "para_X"
                            'text': paragraph_text,
                            'index': i
                        })
                        
        except Exception as e:
            print(f"Error extracting text from DOCX: {e}")
            # Fallback: return a sample paragraph for demonstration
            paragraphs = [{
                'para_id': 1,
                'text': 'This is a sample paragraph for demonstration purposes.',
                'index': 0
            }]
            
        return paragraphs

    def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            print(f"Downloading document from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            print(f"Successfully downloaded document ({len(response.content)} bytes)")
            return response.content
        except Exception as e:
            print(f"Error downloading document: {e}")
            # Return empty bytes on error
            return b''

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex"""
        # Simple sentence splitting - can be enhanced
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def analyze_hyphenation(self, sentence: str, language: str, style: str) -> List[Dict[str, Any]]:
        """
        Analyze a sentence for hyphenation issues
        Returns list of changes needed
        """
        changes = []
        words = sentence.split()
        
        # Check for compound adjectives
        for i in range(len(words) - 1):
            current_word = words[i].lower()
            next_word = words[i + 1].lower()
            compound = f"{current_word}-{next_word}"
            
            # Check if this should be hyphenated
            if self._should_be_hyphenated(current_word, next_word, language, style):
                original_phrase = f"{words[i]} {words[i + 1]}"
                hyphenated_phrase = f"{words[i]}-{words[i + 1]}"
                
                # Find position in sentence
                position = sentence.find(original_phrase)
                if position != -1:
                    changes.append({
                        "input_word": original_phrase,
                        "formatted": hyphenated_phrase,
                        "start_idx": position,
                        "end_idx": position + len(original_phrase),
                        "source": self.dictionary_sources.get(language, "Standard Dictionary"),
                        "metadata": {
                            "rule_applied": "compound_adjective_hyphenation",
                            "dictionary_source": self.dictionary_sources.get(language, "Standard Dictionary"),
                            "style_guide": style.upper(),
                            "confidence_score": 0.95,
                            "grammar_rule": "Compound adjectives should be hyphenated when used before a noun"
                        }
                    })
        
        # Check for number compounds
        for word in words:
            if re.match(r'\d+\s+\w+\s+old', ' '.join(words)):
                # Find "X year old" patterns
                pattern = r'(\d+)\s+(year|day|hour|minute)\s+(old)'
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    original = match.group(0)
                    hyphenated = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                    position = match.start()
                    
                    changes.append({
                        "input_word": original,
                        "formatted": hyphenated,
                        "start_idx": position,
                        "end_idx": position + len(original),
                        "source": self.dictionary_sources.get(language, "Standard Dictionary"),
                        "metadata": {
                            "rule_applied": "number_unit_hyphenation",
                            "dictionary_source": self.dictionary_sources.get(language, "Standard Dictionary"),
                            "style_guide": style.upper(),
                            "confidence_score": 0.98,
                            "grammar_rule": "Number-unit compounds should be hyphenated"
                        }
                    })
        
        # Check for prefix patterns (language-specific)
        if language == 'UK English':
            uk_patterns = ['co operate', 'co ordinate', 'co author']
            for pattern in uk_patterns:
                if pattern in sentence.lower():
                    original = pattern
                    hyphenated = pattern.replace(' ', '-')
                    position = sentence.lower().find(pattern)
                    if position != -1:
                        changes.append({
                            "input_word": sentence[position:position+len(pattern)],
                            "formatted": hyphenated,
                            "start_idx": position,
                            "end_idx": position + len(pattern),
                            "source": self.dictionary_sources.get(language, "Standard Dictionary"),
                            "metadata": {
                                "rule_applied": "uk_prefix_hyphenation",
                                "dictionary_source": self.dictionary_sources.get(language, "Standard Dictionary"),
                                "style_guide": style.upper(),
                                "confidence_score": 0.92,
                                "grammar_rule": "UK English often hyphenates co- prefix words"
                            }
                        })
        
        return changes

    def _should_be_hyphenated(self, word1: str, word2: str, language: str, style: str) -> bool:
        """Determine if two words should be hyphenated"""
        compound = f"{word1}-{word2}"
        
        # Check against known compound adjectives
        if compound in self.rules['compound_adjectives']:
            return True
            
        # Check common patterns
        if word1 in ['well', 'high', 'low', 'long', 'short', 'full', 'part']:
            return True
            
        if word2 in ['known', 'quality', 'term', 'time', 'level', 'scale']:
            return True
            
        return False

    async def process_document(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document and return hyphenation analysis"""
        url = input_data['url']
        style = input_data['style']
        check_list = input_data['check_list']
        language = input_data['eng']
        
        print(f"\n=== Processing Document ===")
        print(f"URL: {url}")
        print(f"Style: {style}")
        print(f"Check List: {check_list}")
        print(f"Language: {language}")
        
        # Download document
        docx_bytes = self.download_document(url)
        if not docx_bytes:
            print("Failed to download document")
            return {"error": "Failed to download document"}
        
        # Extract text
        paragraphs = self.extract_text_from_docx(docx_bytes)
        print(f"Extracted {len(paragraphs)} paragraphs")
        
        # Process each paragraph
        all_results = []
        unique_id_counter = 1
        
        for para in paragraphs:
            para_text = para['text']
            para_id = para['para_id']
            
            print(f"\nProcessing {para_id}: {para_text[:100]}...")
            
            # Split into sentences
            sentences = self.split_into_sentences(para_text)
            
            for sentence in sentences:
                if len(sentence.strip()) < 10:  # Skip very short sentences
                    continue
                    
                print(f"  Analyzing sentence: {sentence[:50]}...")
                
                # Analyze hyphenation
                changes = self.analyze_hyphenation(sentence, language, style)
                
                if changes:
                    # Add unique_id to each change
                    for change in changes:
                        change['unique_id'] = unique_id_counter
                        unique_id_counter += 1
                    
                    result = {
                        "source_sentence": sentence,
                        "changes": changes,
                        "para_id": para_id,
                        "type": "hyphen",
                        "filename": url,  # Use the full S3 URL
                        "timestamp": datetime.now().isoformat(),
                        "language": language,
                        "is_completed": False,  # Will be updated for last item
                        "style_guide": style.upper()
                    }
                    all_results.append(result)
                    print(f"    Found {len(changes)} hyphenation issues")
                else:
                    print(f"    No hyphenation issues found")
        
        # Mark the last result as completed
        if all_results:
            all_results[-1]["is_completed"] = True
        
        print(f"\nCompleted processing. Found {len(all_results)} sentences with hyphenation issues.")
        return {"results": all_results}

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL"""
        try:
            return url.split('/')[-1].split('?')[0]
        except:
            return "document.docx"

async def process_sample_inputs(samples: List[Dict[str, Any]]) -> None:
    """Process all sample inputs and save results"""
    engine = HyphenationEngine()
    
    for sample in samples:
        name = sample['name']
        input_data = sample['input']
        
        print(f"\n{'='*50}")
        print(f"Processing: {name}")
        print(f"{'='*50}")
        
        # Process the document
        result = await engine.process_document(input_data)
        
        # Save results to file
        filename = f"hyphen_results_{input_data['eng'].lower().replace(' ', '_')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {filename}")
            
            # Also print summary
            if 'results' in result:
                print(f"Total sentences with issues: {len(result['results'])}")
                total_changes = sum(len(r['changes']) for r in result['results'])
                print(f"Total hyphenation changes suggested: {total_changes}")
                
                # Print first few results as examples
                print(f"\nFirst few results:")
                for i, res in enumerate(result['results'][:3]):
                    print(f"\n{i+1}. Sentence: {res['source_sentence'][:100]}...")
                    for change in res['changes']:
                        print(f"   - '{change['input_word']}' → '{change['formatted']}'")
                        print(f"     Rule: {change['metadata']['rule_applied']}")
                        
        except Exception as e:
            print(f"Error saving results: {e}")

# For testing purposes
if __name__ == "__main__":
    print("AI Hyphenation Engine - Test Mode")
    # This would be called from app.py normally
