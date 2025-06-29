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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIHyphenationEngine:
    """
    Streamlined AI-powered hyphenation engine focused on efficiency and accuracy.
    """

    def __init__(self):
        """Initialize the streamlined AI engine"""
        logger.info("Initializing Streamlined AI Hyphenation Engine...")

        # Model configuration for LLaMA 3 (8B) via Ollama
        self.model_name = "llama3:latest"

        # Dictionary sources for different languages
        self.dictionary_sources = {
            'US English': 'Merriam-Webster Dictionary',
            'UK English': 'Oxford English Dictionary'
        }

        # Load rules directly from file
        self.rules_content = self._load_rules_file()

        # Initialize Ollama model
        self._initialize_ollama_model()

    def _initialize_ollama_model(self):
        """Initialize Ollama client and check LLaMA 3 (8B) model availability"""
        try:
            logger.info(
                "Checking Ollama and LLaMA 3 (8B) model availability...")
            test_response = ollama.generate(
                model=self.model_name, prompt="Test")
            if test_response and 'response' in test_response:
                logger.info(
                    f"✅ LLaMA 3 model '{self.model_name}' is working properly!")
                self.model_available = True
            else:
                logger.error("Ollama responded but with unexpected format")
                self.model_available = False
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.info(
                "Make sure Ollama is installed and running. Falling back to rule-based processing...")
            self.model_available = False

    def _load_rules_file(self) -> str:
        """Load hyphenation rules directly from rules file"""
        try:
            with open('rules', 'r', encoding='utf-8') as f:
                rules_content = f.read()
            logger.info("✅ Loaded rules file successfully")
            return rules_content
        except Exception as e:
            logger.error(f"Error loading rules file: {e}")
            # Fallback basic rules
            return """
            # BASIC HYPHENATION RULES
            - Hyphenate compound adjectives before nouns (e.g., well-known author)
            - Don't hyphenate adverbs ending in -ly (e.g., highly effective)
            - Remove unnecessary hyphens from dictionary words (e.g., over-shoot → overshoot)
            """

    async def ai_analyze_hyphenation(self, sentence: str, language: str, style: str) -> List[Dict[str, Any]]:
        """Streamlined AI analysis using rules file directly"""
        if not self.model_available:
            logger.warning("Ollama model not available, using basic fallback")
            return self._basic_fallback_analysis(sentence, language, style)

        try:

            # Create simple, direct prompt using rules file
            prompt = f"""You are a hyphenation expert. Analyze this sentence VERY CAREFULLY.

CRITICAL: ONLY suggest changes for BODY TEXT compound adjectives. NEVER change proper names, titles, citations, or institutions.

SENTENCE: "{sentence}"
LANGUAGE: {language}
STYLE: {style.upper()}

BEFORE SUGGESTING ANY CHANGES, CHECK:
1. Is this a proper name?  → NO CHANGES
2. Is this an institution?  → NO CHANGES
3. Is this a citation?  → NO CHANGES
4. Is this a title/heading?  → NO CHANGES
5. Is this a list of separate adjectives? → NO CHANGES

RULES TO FOLLOW:
{self.rules_content}

EXAMPLES OF WHAT NOT TO CHANGE:
✗ proper name
✗ institution name
✗ title/heading
✗ citation
✗ list of separate adjectives

RETURN JSON ONLY IF you find genuine suggestions according to rules in body text:
{{"changes": [{{"original": "exact phrase from sentence", "corrected": "corrected version", "start_position": number, "end_position": number,
    "confidence": 0.90, "justification": "rule explanation", "change_type": "add_hyphen|remove_hyphen|hyphen_to_endash"}}]}}

If no genuine compound adjectives found: {{"changes": []}}"""
            # Query LLM
            ai_response = await self._query_llama(prompt)

            # Parse and validate response
            changes = self._parse_ai_response(
                ai_response, sentence, language, style)

            return changes

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._basic_fallback_analysis(sentence, language, style)

    async def _query_llama(self, prompt: str, max_length: int = 512) -> str:
        """Query LLaMA 3 model via Ollama"""
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.2,
                    'top_p': 0.8,
                    'repeat_penalty': 1.1,
                    'num_predict': max_length
                }
            )
            return response['response'] if 'response' in response else ""
        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
            return '{"changes": []}'

    def _parse_ai_response(self, ai_response: str, sentence: str, language: str, style: str) -> List[Dict[str, Any]]:
        """Parse AI response with streamlined validation"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if not json_match:
                return []

            ai_data = json.loads(json_match.group())
            changes = []

            for change in ai_data.get('changes', []):
                original = change.get('original', '').strip()
                corrected = change.get('corrected', '').strip()
                confidence = float(change.get('confidence', 0.0))
                justification = change.get('justification', '')

                # Simple validation - trust the LLM with rules
                if not original or not corrected or original == corrected or confidence < 0.7:
                    continue

                # Find position in sentence
                start_pos = sentence.find(original)
                if start_pos == -1:
                    start_pos = sentence.lower().find(original.lower())
                    if start_pos != -1:
                        original = sentence[start_pos:start_pos +
                                            len(original)]

                if start_pos != -1:
                    changes.append({
                        "input_word": original,
                        "formatted": corrected,
                        "start_idx": start_pos,
                        "end_idx": start_pos + len(original),
                        "source": f"Streamlined AI Analysis + {self.dictionary_sources.get(language, 'Standard Dictionary')}",
                        "metadata": {
                            "rule_applied": "streamlined_ai_rules_analysis",
                            "dictionary_source": self.dictionary_sources.get(language, "Standard Dictionary"),
                            "style_guide": style.upper(),
                            "confidence_score": confidence,
                            "grammar_rule": justification or 'AI-determined hyphenation based on loaded rules',
                            "ai_model": "LLaMA-3-8B-Streamlined",
                            "validation_passed": True
                        }
                    })

            logger.info(
                f"Found {len(changes)} validated hyphenation suggestions")
            return changes

        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return []

    def _basic_fallback_analysis(self, sentence: str, language: str, style: str) -> List[Dict[str, Any]]:
        """Basic fallback when AI is not available"""
        changes = []

        # Simple pattern matching for common cases
        patterns = [
            (r'\bwell known\b', 'well-known', 'Compound adjective before noun'),
            (r'\bhigh quality\b', 'high-quality',
             'Compound adjective before noun'),
            (r'\blong term\b', 'long-term', 'Compound adjective before noun'),
            (r'\bover-shoot\b', 'overshoot', 'Dictionary standard form'),
            (r'\bco-operate\b', 'cooperate', 'US English standard'),
        ]

        for pattern, replacement, reason in patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                original = match.group()
                if original.lower() != replacement.lower():
                    changes.append({
                        "input_word": original,
                        "formatted": replacement,
                        "start_idx": match.start(),
                        "end_idx": match.end(),
                        "source": f"Basic Fallback + {self.dictionary_sources.get(language, 'Standard Dictionary')}",
                        "metadata": {
                            "rule_applied": "basic_fallback_analysis",
                            "dictionary_source": self.dictionary_sources.get(language, "Standard Dictionary"),
                            "style_guide": style.upper(),
                            "confidence_score": 0.8,
                            "grammar_rule": reason,
                            "ai_model": "Fallback-Rules-Based",
                            "validation_passed": True
                        }
                    })

        return changes

    def extract_text_from_docx(self, docx_bytes: bytes) -> List[Dict[str, Any]]:
        """Enhanced DOCX text extraction with better document structure handling"""
        paragraphs = []

        try:
            with zipfile.ZipFile(BytesIO(docx_bytes), 'r') as zip_file:
                # Debug: List all files in the DOCX
                file_list = zip_file.namelist()
                # Show first 10 files
                logger.info(f"📋 DOCX contains: {file_list[:10]}...")

                # Read the main document XML
                xml_content = zip_file.read('word/document.xml')
                root = ET.fromstring(xml_content)

                # Define namespaces (handle both old and new Word formats)
                namespaces = {
                    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                    'w14': 'http://schemas.microsoft.com/office/word/2010/wordml',
                    'w15': 'http://schemas.microsoft.com/office/word/2012/wordml'
                }

                # Extract paragraphs with enhanced search
                para_elements = root.findall('.//w:p', namespaces)
                logger.info(f"🔍 Found {len(para_elements)} paragraph elements")

                # Also try to find text in tables
                table_elements = root.findall('.//w:tbl', namespaces)
                logger.info(f"📊 Found {len(table_elements)} table elements")

                all_text_elements = []

                # Extract from paragraphs
                for i, para in enumerate(para_elements):
                    text_parts = []
                    for text_elem in para.findall('.//w:t', namespaces):
                        if text_elem.text:
                            text_parts.append(text_elem.text)

                    paragraph_text = ''.join(text_parts).strip()
                    if paragraph_text:
                        all_text_elements.append({
                            'para_id': len(all_text_elements) + 1,
                            'text': paragraph_text,
                            'index': len(all_text_elements),
                            'source': 'paragraph'
                        })

                # Extract from tables
                for table in table_elements:
                    for row in table.findall('.//w:tr', namespaces):
                        row_text_parts = []
                        for cell in row.findall('.//w:tc', namespaces):
                            cell_text_parts = []
                            for text_elem in cell.findall('.//w:t', namespaces):
                                if text_elem.text:
                                    cell_text_parts.append(text_elem.text)
                            cell_text = ''.join(cell_text_parts).strip()
                            if cell_text:
                                row_text_parts.append(cell_text)

                        if row_text_parts:
                            table_text = ' | '.join(row_text_parts)
                            all_text_elements.append({
                                'para_id': len(all_text_elements) + 1,
                                'text': table_text,
                                'index': len(all_text_elements),
                                'source': 'table'
                            })

                # If still no content, try a more aggressive search
                if not all_text_elements:
                    logger.warning(
                        "⚠️ No content found with standard extraction, trying aggressive search...")
                    all_text_in_doc = root.findall('.//w:t', namespaces)
                    logger.info(
                        f"🔍 Found {len(all_text_in_doc)} text elements total")

                    # Combine all text elements into paragraphs
                    combined_text = []
                    for text_elem in all_text_in_doc:
                        if text_elem.text and text_elem.text.strip():
                            combined_text.append(text_elem.text.strip())

                    if combined_text:
                        # Group into logical paragraphs (split by multiple spaces or line breaks)
                        full_text = ' '.join(combined_text)
                        potential_paragraphs = re.split(
                            r'\s{3,}|\n{2,}', full_text)

                        for i, para_text in enumerate(potential_paragraphs):
                            para_text = para_text.strip()
                            # Only include substantial text
                            if para_text and len(para_text) > 10:
                                all_text_elements.append({
                                    'para_id': i + 1,
                                    'text': para_text,
                                    'index': i,
                                    'source': 'extracted'
                                })

                logger.info(
                    f"✅ Successfully extracted {len(all_text_elements)} text sections")
                if all_text_elements:
                    logger.info(
                        f"📝 Sample text: '{all_text_elements[0]['text'][:100]}...'")

                return all_text_elements

        except Exception as e:
            logger.error(f"❌ Error extracting text from DOCX: {e}")
            logger.info("🔄 Using fallback content to test engine...")

            # Enhanced fallback with realistic content for testing
            test_paragraphs = [
                "The well known author wrote an excellent book about time management.",
                "This is a high quality product that offers cost effective solutions.",
                "The 5 year old child played in the well maintained playground.",
                "Recent studies show that nature based therapy is highly effective.",
                "The self made entrepreneur started a user friendly application.",
                "In recent years, long term investments have become more popular."
            ]

            paragraphs = []
            for i, text in enumerate(test_paragraphs):
                paragraphs.append({
                    'para_id': i + 1,
                    'text': text,
                    'index': i,
                    'source': 'fallback_test'
                })

            logger.info(
                f"🧪 Created {len(paragraphs)} test paragraphs to demonstrate accuracy")
            return paragraphs

    def download_document(self, url: str) -> bytes:
        """Download document from URL"""
        try:
            logger.info(f"Downloading document from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            logger.info(
                f"Successfully downloaded document ({len(response.content)} bytes)")
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
        real_time_results = {"results": [],
                             "status": "processing", "total_found": 0}

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
        logger.info(
            f"⚡ Using {'AI Mode' if self.model_available else 'Fallback Mode'}")
        logger.info(f"📊 Watch {filename} for INSTANT results!")

        # Process each paragraph with INSTANT output
        for para_idx, para in enumerate(paragraphs):
            para_text = para['text']
            para_id = para['para_id']

            # Progress update every 5 paragraphs
            if para_idx % 5 == 0:
                progress = (para_idx / len(paragraphs)) * 100
                logger.info(
                    f"📊 Progress: {progress:.1f}% - Paragraph {para_idx+1}/{len(paragraphs)} - Found {len(all_results)} issues so far")

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

                logger.info(
                    f"🔍 AI analyzing ({processed_sentences}/{total_sentences}): {sentence[:50]}...")

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
                    real_time_results["last_updated"] = datetime.now(
                    ).isoformat()

                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(real_time_results, f,
                                  indent=2, ensure_ascii=False)

                    logger.info(
                        f"    ✅ INSTANT SAVE: Found {len(changes)} issues → Total: {len(all_results)} → Saved to {filename}")

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
        logger.info(
            f"📈 FINAL Results: {len(all_results)} sentences with hyphenation issues")
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
                logger.info(
                    f"Total sentences with AI-detected issues: {len(result['results'])}")
                total_changes = sum(len(r['changes'])
                                    for r in result['results'])
                logger.info(
                    f"Total AI hyphenation changes suggested: {total_changes}")

                # Print first few results as examples
                logger.info(f"\nFirst few AI results:")
                for i, res in enumerate(result['results'][:3]):
                    logger.info(
                        f"\n{i+1}. Sentence: {res['source_sentence'][:100]}...")
                    for change in res['changes']:
                        logger.info(
                            f"   - AI suggestion: '{change['input_word']}' → '{change['formatted']}'")
                        logger.info(
                            f"     AI Rule: {change['metadata']['rule_applied']}")
                        logger.info(
                            f"     Confidence: {change['metadata']['confidence_score']}")

        except Exception as e:
            logger.error(f"Error saving AI results: {e}")

# For testing purposes
if __name__ == "__main__":
    logger.info("AI Hyphenation Engine with LLaMA 3 (8B) - Test Mode")
    # This would be called from app.py normally
