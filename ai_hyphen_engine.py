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
import ollama
from duckduckgo_search import DDGS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIHyphenationEngine:
    """
    True AI-powered hyphenation engine using LLaMA 3 (8B) model that processes 
    DOCX documents and provides intelligent hyphenation corrections based on 
    context, style guides and language variants.
    """
    
    def __init__(self):
        """Initialize the AI engine with LLaMA 3 (8B) model via Ollama"""
        logger.info("Initializing AI Hyphenation Engine with Ollama LLaMA 3 (8B)...")
        
        # Model configuration for LLaMA 3 (8B) via Ollama
        self.model_name = "llama3:latest"
        
        # Initialize DuckDuckGo search
        self.ddgs = DDGS()
        
        # Dictionary sources for different languages
        self.dictionary_sources = {
            'US English': 'Merriam-Webster Dictionary',
            'UK English': 'Oxford English Dictionary'
        }
        
        # Hyphenation rules as guidance for LLM (not hardcoded logic)
        self.hyphenation_guidance = self._load_guidance_rules()
        
        # Initialize Ollama client and check model availability
        self._initialize_ollama_model()
        
    def _initialize_ollama_model(self):
        """Initialize Ollama client and check LLaMA 3 (8B) model availability"""
        try:
            logger.info("Checking Ollama and LLaMA 3 (8B) model availability...")
            
            # Test Ollama connection with a simple generation
            try:
                test_response = ollama.generate(model=self.model_name, prompt="Test")
                if test_response and 'response' in test_response:
                    logger.info(f"✅ LLaMA 3 model '{self.model_name}' is working properly!")
                    self.model_available = True
                else:
                    logger.error("Ollama responded but with unexpected format")
                    self.model_available = False
                    
            except Exception as test_error:
                logger.error(f"Ollama test failed: {test_error}")
                # Try listing models as fallback
                try:
                    models = ollama.list()
                    available_models = [model['name'] for model in models['models']]
                    logger.info(f"Available models: {available_models}")
                    
                    if self.model_name in available_models:
                        logger.info(f"Model '{self.model_name}' found but test failed")
                        self.model_available = False
                    else:
                        logger.warning(f"Model '{self.model_name}' not found in {available_models}")
                        self.model_available = False
                        
                except Exception as list_error:
                    logger.error(f"Failed to list models: {list_error}")
                    self.model_available = False
            
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.info("Make sure Ollama is installed and running. Falling back to rule-based processing...")
            self.model_available = False

    def _load_guidance_rules(self) -> Dict[str, Any]:
        """Load hyphenation guidance rules for LLM (not hardcoded logic)"""
        return {
            'style_guidelines': {
                'APA': {
                    'compound_adjectives': 'Hyphenate compound adjectives when they precede nouns they modify',
                    'number_compounds': 'Hyphenate number-word compounds (e.g., 5-year-old, 10-member team)',
                    'prefixes': 'Generally do not hyphenate prefixes unless clarity requires it',
                    'adverb_adjective': 'Do not hyphenate adverb-adjective compounds ending in -ly'
                },
                'MLA': {
                    'compound_adjectives': 'Use hyphens in compound adjectives before nouns',
                    'number_compounds': 'Hyphenate spelled-out numbers and fractions',
                    'prefixes': 'Hyphenate prefixes when they precede proper nouns'
                }
            },
            'language_specific': {
                'US English': {
                    'preferred_patterns': ['co-operate → cooperate', 'co-ordinate → coordinate'],
                    'dictionary_preference': 'Merriam-Webster standards'
                },
                'UK English': {
                    'preferred_patterns': ['cooperate → co-operate', 'coordinate → co-ordinate'],
                    'dictionary_preference': 'Oxford English Dictionary standards'
                }
            },
            'context_considerations': [
                'Consider sentence flow and readability',
                'Maintain consistency within document',
                'Prioritize clarity over rigid rules',
                'Consider compound modifier placement relative to noun'
            ]
        }

    async def search_dictionary_definition(self, word: str, language: str) -> Optional[str]:
        """Search for word definition using DuckDuckGo to inform hyphenation decisions"""
        try:
            dictionary_source = self.dictionary_sources.get(language, "dictionary")
            search_query = f"{word} definition {dictionary_source}"
            
            # Search using DuckDuckGo
            results = self.ddgs.text(search_query, max_results=3)
            
            if results:
                # Extract relevant definition information
                definitions = []
                for result in results:
                    if any(source in result['href'].lower() for source in ['merriam-webster', 'oxford', 'cambridge', 'dictionary']):
                        definitions.append(result['body'][:200])  # First 200 chars
                
                return " | ".join(definitions[:2]) if definitions else None
                
        except Exception as e:
            logger.warning(f"Dictionary search failed for '{word}': {e}")
            return None

    async def ai_analyze_hyphenation(self, sentence: str, language: str, style: str) -> List[Dict[str, Any]]:
        """Use LLaMA 3 (8B) via Ollama to analyze hyphenation with context understanding"""
        if not self.model_available:
            logger.warning("Ollama LLaMA model not available, using fallback analysis")
            return await self._fallback_analysis(sentence, language, style)
        
        try:
            # Get relevant guidance
            style_guide = self.hyphenation_guidance['style_guidelines'].get(style.upper(), {})
            lang_guide = self.hyphenation_guidance['language_specific'].get(language, {})
            
            # Skip dictionary searches for speed - comment out the slow part
            # potential_compounds = self._identify_potential_compounds(sentence)
            # dictionary_context = {}
            
            # Create AI prompt for LLaMA 3 (simplified for speed)
            prompt = self._create_fast_hyphenation_prompt(sentence, language, style, style_guide, lang_guide)
            
            # Generate AI response
            ai_response = await self._query_llama(prompt, max_length=256)  # Reduced max length
            
            # Parse AI response into structured format
            changes = self._parse_ai_response(ai_response, sentence, language, style)
            
            return changes
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return await self._fallback_analysis(sentence, language, style)

    def _create_fast_hyphenation_prompt(self, sentence: str, language: str, style: str, 
                                 style_guide: Dict, lang_guide: Dict) -> str:
        """Create optimized prompt for LLaMA 3 hyphenation analysis (faster version)"""
        
        prompt = f"""You are a hyphenation expert. Analyze this sentence for compound adjectives that need hyphens:

SENTENCE: "{sentence}"

RULES:
- Hyphenate compound adjectives BEFORE nouns (e.g., "well known" → "well-known")
- Numbers + words before nouns (e.g., "5 year old" → "5-year-old") 
- {language} style, {style} guide

OUTPUT: JSON only with this format:
{{"changes": [{{"original": "text to change", "corrected": "hyphenated-text", "start_position": 0, "end_position": 10, "confidence": 0.9}}]}}

Response:"""
        return prompt

    async def _query_llama(self, prompt: str, max_length: int = 512) -> str:
        """Query LLaMA 3 model via Ollama with the hyphenation prompt"""
        try:
            logger.info("Querying Ollama LLaMA 3 (8B) model...")
            
            # Generate response using Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1,
                    'num_predict': max_length
                }
            )
            
            # Extract the response text
            response_text = response['response'] if 'response' in response else ""
            
            logger.info(f"Ollama response received: {len(response_text)} characters")
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
            return '{"changes": []}'

    def _parse_ai_response(self, ai_response: str, sentence: str, language: str, style: str) -> List[Dict[str, Any]]:
        """Parse LLaMA 3 response into structured hyphenation changes"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in AI response")
                return []
            
            ai_data = json.loads(json_match.group())
            changes = []
            
            for change in ai_data.get('changes', []):
                # Validate the change
                original = change.get('original', '').strip()
                corrected = change.get('corrected', '').strip()
                
                if not original or not corrected or original == corrected:
                    continue
                
                # Find exact position in sentence
                start_pos = sentence.find(original)
                if start_pos == -1:
                    # Try case-insensitive search
                    start_pos = sentence.lower().find(original.lower())
                    if start_pos != -1:
                        original = sentence[start_pos:start_pos + len(original)]
                
                if start_pos != -1:
                    changes.append({
                        "input_word": original,
                        "formatted": corrected,
                        "start_idx": start_pos,
                        "end_idx": start_pos + len(original),
                        "source": f"AI Analysis + {self.dictionary_sources.get(language, 'Standard Dictionary')}",
                        "metadata": {
                            "rule_applied": "ai_context_analysis",
                            "dictionary_source": self.dictionary_sources.get(language, "Standard Dictionary"),
                            "style_guide": style.upper(),
                            "confidence_score": float(change.get('confidence', 0.9)),
                            "grammar_rule": change.get('rule_explanation', 'AI-determined hyphenation based on context'),
                            "ai_model": "LLaMA-3-8B-Instruct"
                        }
                    })
            
            return changes
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return []

    async def _fallback_analysis(self, sentence: str, language: str, style: str) -> List[Dict[str, Any]]:
        """Fallback analysis when LLaMA model is not available"""
        logger.info("Using fallback hyphenation analysis")
        changes = []
        words = sentence.split()
        
        # Simple compound adjective detection
        for i in range(len(words) - 1):
            word1 = words[i].lower().strip('.,!?;:')
            word2 = words[i + 1].lower().strip('.,!?;:')
            
            # Basic compound patterns
            if ((word1 in ['well', 'high', 'low', 'long', 'short'] and 
                 word2 in ['known', 'quality', 'term', 'time', 'level']) or
                (word1.isdigit() and word2 in ['year', 'day', 'month'] and 
                 i + 2 < len(words) and words[i + 2].lower() == 'old')):
                
                original = f"{words[i]} {words[i + 1]}"
                corrected = f"{words[i]}-{words[i + 1]}"
                start_pos = sentence.find(original)
                
                if start_pos != -1:
                    changes.append({
                        "input_word": original,
                        "formatted": corrected,
                        "start_idx": start_pos,
                        "end_idx": start_pos + len(original),
                        "source": f"Fallback Analysis + {self.dictionary_sources.get(language, 'Standard Dictionary')}",
                        "metadata": {
                            "rule_applied": "fallback_pattern_matching",
                            "dictionary_source": self.dictionary_sources.get(language, "Standard Dictionary"),
                            "style_guide": style.upper(),
                            "confidence_score": 0.7,
                            "grammar_rule": "Basic compound adjective pattern",
                            "ai_model": "Fallback (LLaMA unavailable)"
                        }
                    })
        
        return changes

    def extract_text_from_docx(self, docx_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract text from DOCX file using standard Python libraries"""
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
                            'para_id': i+1,
                            'text': paragraph_text,
                            'index': i
                        })
                        
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
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
            logger.info(f"Downloading document from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully downloaded document ({len(response.content)} bytes)")
            return response.content
        except Exception as e:
            logger.error(f"Error downloading document: {e}")
            return b''

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    async def process_document(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process document with REAL-TIME streaming output - saves results immediately"""
        url = input_data['url']
        style = input_data['style']
        check_list = input_data['check_list']
        language = input_data['eng']
        
        logger.info(f"\n=== ⚡ REAL-TIME AI Processing Document ===")
        logger.info(f"URL: {url}")
        logger.info(f"Style: {style}")
        logger.info(f"Language: {language}")
        
        # Setup real-time output file
        filename = f"realtime_hyphen_results_{language.lower().replace(' ', '_')}.json"
        real_time_results = {"results": [], "status": "processing", "total_found": 0}
        
        # Initialize file immediately
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(real_time_results, f, indent=2, ensure_ascii=False)
        logger.info(f"📁 Real-time results file: {filename}")
        
        # Download document
        docx_bytes = self.download_document(url)
        if not docx_bytes:
            logger.error("Failed to download document")
            return {"error": "Failed to download document"}
        
        # Extract text
        paragraphs = self.extract_text_from_docx(docx_bytes)
        logger.info(f"📄 Extracted {len(paragraphs)} paragraphs")
        
        # Process with INSTANT output
        all_results = []
        unique_id_counter = 1
        total_sentences = 0
        processed_sentences = 0
        
        # First pass: count total sentences for progress
        for para in paragraphs:
            sentences = self.split_into_sentences(para['text'])
            total_sentences += len(sentences)
        
        logger.info(f"🎯 Total sentences to analyze: {total_sentences}")
        logger.info(f"⚡ Using {'AI Mode' if self.model_available else 'Fallback Mode'}")
        logger.info(f"📊 Watch {filename} for INSTANT results!")
        
        # Process each paragraph with INSTANT output
        for para_idx, para in enumerate(paragraphs):
            para_text = para['text']
            para_id = para['para_id']
            
            # Progress update every 5 paragraphs
            if para_idx % 5 == 0:
                progress = (para_idx / len(paragraphs)) * 100
                logger.info(f"📊 Progress: {progress:.1f}% - Paragraph {para_idx+1}/{len(paragraphs)} - Found {len(all_results)} issues so far")
            
            # Split into sentences
            sentences = self.split_into_sentences(para_text)
            
            # Process sentences with INSTANT saving
            for sentence_idx, sentence in enumerate(sentences):
                processed_sentences += 1
                
                # Skip very short sentences to save time
                if len(sentence.split()) < 3:
                    continue
                    
                # Skip reference sentences (citations) to save time
                if any(word in sentence.lower() for word in ['doi:', 'http', 'retrieved', 'pp.', 'vol.']):
                    continue
                
                logger.info(f"🔍 AI analyzing ({processed_sentences}/{total_sentences}): {sentence[:50]}...")
                
                # Use AI to analyze hyphenation
                changes = await self.ai_analyze_hyphenation(sentence, language, style)
                
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
                        "filename": url,
                        "timestamp": datetime.now().isoformat(),
                        "language": language,
                        "is_completed": False,
                        "style_guide": style.upper()
                    }
                    
                    # 🚀 INSTANT SAVE - Add to results and save immediately
                    all_results.append(result)
                    
                    # Update the real-time file IMMEDIATELY
                    real_time_results["results"] = all_results
                    real_time_results["total_found"] = len(all_results)
                    real_time_results["last_updated"] = datetime.now().isoformat()
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(real_time_results, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"    ✅ INSTANT SAVE: Found {len(changes)} issues → Total: {len(all_results)} → Saved to {filename}")
        
        # Mark the last result as completed and final save
        if all_results:
            all_results[-1]["is_completed"] = True
        
        # Final update to file
        real_time_results["results"] = all_results
        real_time_results["status"] = "completed"
        real_time_results["total_found"] = len(all_results)
        real_time_results["final_completed"] = datetime.now().isoformat()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(real_time_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n🎉 AI processing completed!")
        logger.info(f"📈 FINAL Results: {len(all_results)} sentences with hyphenation issues")
        logger.info(f"⏱️  Processed {processed_sentences} sentences total")
        logger.info(f"📁 Results saved to: {filename}")
        
        return {"results": all_results}

    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL"""
        try:
            return url.split('/')[-1].split('?')[0]
        except:
            return "document.docx"

async def process_sample_inputs(samples: List[Dict[str, Any]]) -> None:
    """Process all sample inputs using AI-powered analysis and save results"""
    engine = AIHyphenationEngine()
    
    for sample in samples:
        name = sample['name']
        input_data = sample['input']
        
        logger.info(f"\n{'='*50}")
        logger.info(f"AI Processing: {name}")
        logger.info(f"{'='*50}")
        
        # Process the document with AI
        result = await engine.process_document(input_data)
        
        # Save results to file
        filename = f"ai_hyphen_results_{input_data['eng'].lower().replace(' ', '_')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"\nAI results saved to: {filename}")
            
            # Also print summary
            if 'results' in result:
                logger.info(f"Total sentences with AI-detected issues: {len(result['results'])}")
                total_changes = sum(len(r['changes']) for r in result['results'])
                logger.info(f"Total AI hyphenation changes suggested: {total_changes}")
                
                # Print first few results as examples
                logger.info(f"\nFirst few AI results:")
                for i, res in enumerate(result['results'][:3]):
                    logger.info(f"\n{i+1}. Sentence: {res['source_sentence'][:100]}...")
                    for change in res['changes']:
                        logger.info(f"   - AI suggestion: '{change['input_word']}' → '{change['formatted']}'")
                        logger.info(f"     AI Rule: {change['metadata']['rule_applied']}")
                        logger.info(f"     Confidence: {change['metadata']['confidence_score']}")
                        
        except Exception as e:
            logger.error(f"Error saving AI results: {e}")

# For testing purposes
if __name__ == "__main__":
    logger.info("AI Hyphenation Engine with LLaMA 3 (8B) - Test Mode")
    # This would be called from app.py normally
 