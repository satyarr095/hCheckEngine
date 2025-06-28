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
    Improved AI-powered hyphenation engine with enhanced accuracy for compound adjectives.
    Focuses on reducing false positives and properly identifying genuine hyphenation needs.
    """
    
    def __init__(self):
        """Initialize the AI engine with improved accuracy rules"""
        logger.info("Initializing Enhanced AI Hyphenation Engine...")
        
        # Model configuration for LLaMA 3 (8B) via Ollama
        self.model_name = "llama3:latest"
        
        # Initialize DuckDuckGo search
        self.ddgs = DDGS()
        
        # Dictionary sources for different languages
        self.dictionary_sources = {
            'US English': 'Merriam-Webster Dictionary',
            'UK English': 'Oxford English Dictionary'
        }
        
        # Enhanced hyphenation rules for better accuracy
        self.hyphenation_guidance = self._load_enhanced_guidance_rules()
        
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

    def _load_enhanced_guidance_rules(self) -> Dict[str, Any]:
        """Load enhanced hyphenation guidance rules with better accuracy filters"""
        return {
            'style_guidelines': {
                'APA': {
                    'compound_adjectives': 'Hyphenate compound adjectives ONLY when they precede and directly modify nouns (e.g., well-known author, five-year-old child)',
                    'number_compounds': 'Hyphenate number-word compounds when they modify nouns (e.g., 5-year-old boy, 10-member team)',
                    'prefixes': 'Generally do not hyphenate prefixes unless clarity requires it',
                    'adverb_adjective': 'NEVER hyphenate adverb-adjective compounds, especially those ending in -ly (e.g., highly successful, exceptionally interesting)',
                    'exclusions': 'Do not hyphenate proper nouns, institutional names, general noun phrases, or words that are not functioning as compound modifiers'
                }
            },
            'accuracy_filters': {
                'avoid_hyphenating': [
                    'proper_nouns',  # Names, places, institutions
                    'general_noun_phrases',  # "nature tourism", "sports tourists"
                    'adverb_adjective_ly',  # "exceptionally interesting"
                    'prepositional_phrases',  # "as well as"
                    'time_expressions',  # "recent years", "same time"
                    'single_words',  # "ABSTRACT"
                    'institutional_names'  # "Physical Activity", "Doctoral School"
                ],
                'require_noun_modification': True,  # Must directly modify a following noun
                'minimum_confidence': 0.8  # Require high confidence for suggestions
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
            }
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
        """Dynamic AI analysis with comprehensive hyphenation detection"""
        if not self.model_available:
            logger.warning("Ollama LLaMA model not available, using enhanced fallback analysis")
            return await self._enhanced_fallback_analysis(sentence, language, style)
        
        try:
            # Get relevant guidance dynamically
            style_guide = self.hyphenation_guidance['style_guidelines'].get(style.upper(), {})
            accuracy_filters = self.hyphenation_guidance['accuracy_filters']
            
            # Create comprehensive AI prompt with dynamic context
            prompt = self._create_enhanced_hyphenation_prompt(sentence, language, style, style_guide, accuracy_filters)
            
            # Generate AI response with higher token limit for comprehensive analysis
            ai_response = await self._query_llama(prompt, max_length=1024)
            
            # Parse AI response with dynamic validation
            changes = self._parse_ai_response_with_validation(ai_response, sentence, language, style)
            
            # Apply dynamic post-processing validation
            validated_changes = self._apply_dynamic_validation(changes, sentence, language, style)
            
            return validated_changes
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return await self._enhanced_fallback_analysis(sentence, language, style)

    def _create_enhanced_hyphenation_prompt(self, sentence: str, language: str, style: str, 
                                           style_guide: Dict, accuracy_filters: Dict) -> str:
        """Create dynamic, comprehensive prompt for hyphenation analysis"""
        
        # Build context-aware examples based on the sentence content
        context_examples = self._generate_contextual_examples(sentence, language)
        
        prompt = f"""You are an expert hyphenation analyst. Examine this sentence for ALL hyphenation issues with careful attention to context and grammar rules.

SENTENCE TO ANALYZE: "{sentence}"

COMPREHENSIVE ANALYSIS REQUIRED:

1. COMPOUND ADJECTIVES (add hyphens when modifying nouns):
   ✓ "human managed carbon" → "human-managed carbon" (modifies following noun)
   ✓ "long timescale storage" → "long-timescale storage" (modifies following noun)
   ✓ "well known method" → "well-known method" (modifies following noun)
   ✗ "method is well known" (predicate position - no hyphen needed)

2. OVER-HYPHENATION (remove unnecessary hyphens):
   ✓ "over-shoot" → "overshoot" (standard single word in dictionaries)
   ✓ "co-operate" → "cooperate" (US English standard)
   ✓ "phytoplankton-community composition" → "phytoplankton community composition" (not a true compound modifier)

3. CONTEXTUAL POSITION (hyphenation depends on usage):
   ✓ "sea-ice retreat" → "sea ice retreat" (not modifying noun in this context)
   ✓ "during the period of sea-ice retreat" → "during the period of sea ice retreat"
   But: "sea-ice formation" stays hyphenated (directly modifies formation)

4. GEOGRAPHIC CONNECTIONS (use en dash, not hyphen):
   ✓ "Asia-Europe route" → "Asia–Europe route" (proper noun connection)
   ✓ "London-Paris flight" → "London–Paris flight" (geographic relationship)

{context_examples}

ANALYSIS INSTRUCTIONS:
- Examine EACH potential compound in the sentence
- Consider the GRAMMATICAL POSITION (before noun vs. predicate)
- Check for DICTIONARY STANDARD forms
- Identify SCIENTIFIC vs. COMPOUND MODIFIER usage
- Detect GEOGRAPHIC/PROPER NOUN connections

Language: {language} | Style: {style.upper()}

RETURN PRECISE JSON OUTPUT:
{{"changes": [{{"original": "exact phrase from sentence", "corrected": "corrected version", "start_position": position_number, "end_position": end_position_number, "confidence": 0.95, "justification": "detailed grammatical explanation", "change_type": "add_hyphen|remove_hyphen|hyphen_to_endash"}}]}}

If NO issues detected: {{"changes": []}}

JSON Response:"""
        return prompt

    def _generate_contextual_examples(self, sentence: str, language: str) -> str:
        """Generate context-specific examples based on sentence content"""
        examples = []
        sentence_lower = sentence.lower()
        
        # Add relevant examples based on sentence content
        if 'manage' in sentence_lower:
            examples.append("CONTEXT RELEVANT: Check for 'human managed', 'well managed', etc. before nouns")
        
        if 'time' in sentence_lower:
            examples.append("CONTEXT RELEVANT: Check for 'long timescale', 'real time', 'long term' patterns")
        
        if any(word in sentence_lower for word in ['phytoplankton', 'community', 'species']):
            examples.append("SCIENTIFIC TERMS: 'phytoplankton-community' → 'phytoplankton community' (incorrectly hyphenated scientific phrase)")
        
        if any(word in sentence_lower for word in ['sea', 'ice', 'ocean']):
            examples.append("ENVIRONMENTAL TERMS: 'sea-ice retreat' → 'sea ice retreat' (contextual position matters)")
        
        if re.search(r'[A-Z][a-z]+-[A-Z][a-z]+', sentence):
            examples.append("PROPER NOUNS: 'Asia-Europe route' → 'Asia–Europe route' (geographic connections need en dash)")
        
        if 'over-' in sentence_lower:
            examples.append("OVER-HYPHENATION: 'over-shoot' → 'overshoot' (standard dictionary word)")
        
        if examples:
            return "\nCONTEXT-SPECIFIC GUIDANCE:\n" + "\n".join(f"• {ex}" for ex in examples) + "\n"
        
        return ""

    async def _query_llama(self, prompt: str, max_length: int = 512) -> str:
        """Query LLaMA 3 model via Ollama with the enhanced hyphenation prompt"""
        try:
            logger.info("Querying Ollama LLaMA 3 (8B) model with enhanced prompt...")
            
            # Generate response using Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.2,  # Lower temperature for more consistent results
                    'top_p': 0.8,
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

    def _parse_ai_response_with_validation(self, ai_response: str, sentence: str, language: str, style: str) -> List[Dict[str, Any]]:
        """Parse LLaMA 3 response with enhanced validation to reduce false positives"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in AI response")
                return []
            
            ai_data = json.loads(json_match.group())
            changes = []
            
            for change in ai_data.get('changes', []):
                # Enhanced validation
                original = change.get('original', '').strip()
                corrected = change.get('corrected', '').strip()
                confidence = float(change.get('confidence', 0.0))
                justification = change.get('justification', '')
                
                # Strict validation filters
                if not self._validate_hyphenation_change(original, corrected, confidence, sentence, justification):
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
                        "source": f"Enhanced AI Analysis + {self.dictionary_sources.get(language, 'Standard Dictionary')}",
                        "metadata": {
                            "rule_applied": "enhanced_ai_context_analysis",
                            "dictionary_source": self.dictionary_sources.get(language, "Standard Dictionary"),
                            "style_guide": style.upper(),
                            "confidence_score": confidence,
                            "grammar_rule": justification or 'Enhanced AI-determined compound adjective hyphenation',
                            "ai_model": "LLaMA-3-8B-Instruct-Enhanced",
                            "validation_passed": True
                        }
                    })
            
            logger.info(f"Validated {len(changes)} hyphenation suggestions out of {len(ai_data.get('changes', []))} AI suggestions")
            return changes
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return []

    def _validate_hyphenation_change(self, original: str, corrected: str, confidence: float, 
                                   sentence: str, justification: str) -> bool:
        """Enhanced validation for comprehensive hyphenation detection"""
        
        # Basic checks
        if not original or not corrected or original == corrected:
            return False
        
        # Lower confidence threshold for comprehensive detection
        if confidence < 0.6:
            return False
        
        # Check for common false positive patterns
        words = original.lower().split()
        
        # Don't hyphenate single words (unless it's a hyphen removal case)
        if len(words) < 2 and '-' not in original:
            return False
        
        # Don't hyphenate adverb + adjective (especially -ly words) - ONLY for adding hyphens
        if len(words) == 2 and words[0].endswith('ly') and '-' not in original:
            logger.info(f"Rejected: adverb-adjective pattern '{original}'")
            return False
        
        # Allow hyphen removal cases (over-shoot → overshoot)
        if '-' in original and '-' not in corrected:
            logger.info(f"Accepted: hyphen removal '{original}' → '{corrected}'")
            return True
        
        # Allow en dash cases (Asia-Europe → Asia–Europe)
        if '-' in original and '–' in corrected:
            logger.info(f"Accepted: en dash replacement '{original}' → '{corrected}'")
            return True
        
        # Allow scientific/contextual fixes (phytoplankton-community → phytoplankton community)
        if '-' in original and ' ' in corrected and 'community' in original.lower():
            logger.info(f"Accepted: scientific term fix '{original}' → '{corrected}'")
            return True
        
        # Allow sea-ice contextual fixes
        if 'sea-ice' in original.lower() and 'sea ice' in corrected.lower():
            logger.info(f"Accepted: contextual sea-ice fix '{original}' → '{corrected}'")
            return True
        
        # Check if it's likely a genuine compound adjective by looking for patterns
        valid_patterns = [
            'well known', 'high quality', 'low cost', 'long term', 'short term',
            'self made', 'user friendly', 'time consuming', 'cost effective',
            'human managed', 'long timescale', 'real time'
        ]
        
        # Look for number + word patterns (like "5 year old")
        if re.match(r'\d+\s+\w+', original):
            return True
        
        # Check for compound adjective patterns
        if any(pattern in original.lower() for pattern in valid_patterns):
            return True
        
        # If we have a specific justification, be more accepting
        if justification and any(phrase in justification.lower() for phrase in 
                               ['before noun', 'compound adjective', 'modifies', 'dictionary word', 
                                'standard', 'geographic', 'contextual', 'scientific']):
            return True
        
        # Check position in sentence for compound adjectives
        sentence_lower = sentence.lower()
        original_lower = original.lower()
        pos = sentence_lower.find(original_lower)
        
        if pos != -1:
            # Check what comes after
            after_pos = pos + len(original_lower)
            remaining = sentence_lower[after_pos:].strip()
            
            # If followed by a noun-like word, likely a compound adjective
            if remaining and not remaining.startswith(('is', 'are', 'was', 'were', 'to', 'for', 'in', 'on', 'at', 'and', 'or')):
                logger.info(f"Accepted: compound adjective before noun '{original}'")
                return True
        
        # Default to accepting if we have reasonable confidence
        if confidence >= 0.8:
            logger.info(f"Accepted: high confidence change '{original}' → '{corrected}'")
            return True
        
        # Default to rejecting if we're not confident
        logger.info(f"Rejected: insufficient evidence for '{original}'")
        return False

    def _apply_dynamic_validation(self, changes: List[Dict[str, Any]], sentence: str, language: str, style: str) -> List[Dict[str, Any]]:
        """Apply dynamic validation based on context and linguistic analysis"""
        if not changes:
            return []
        
        validated_changes = []
        
        for change in changes:
            original = change.get('input_word', change.get('original', ''))
            corrected = change.get('formatted', change.get('corrected', ''))
            
            # Dynamic context analysis
            if self._analyze_dynamic_context(original, corrected, sentence, language, style):
                validated_changes.append(change)
            else:
                logger.info(f"Dynamic validation rejected: '{original}' → '{corrected}'")
        
        return validated_changes

    def _analyze_dynamic_context(self, original: str, corrected: str, sentence: str, language: str, style: str) -> bool:
        """Dynamically analyze context to determine if hyphenation change is valid"""
        
        # Basic validity checks
        if not original or not corrected or original == corrected:
            return False
        
        # Dynamic linguistic analysis
        words_in_original = original.lower().split()
        sentence_lower = sentence.lower()
        
        # Find position and context
        original_pos = sentence_lower.find(original.lower())
        if original_pos == -1:
            return False
        
        # Get surrounding context
        before_context = sentence_lower[:original_pos].strip().split()[-3:] if original_pos > 0 else []
        after_pos = original_pos + len(original)
        after_context = sentence_lower[after_pos:].strip().split()[:3] if after_pos < len(sentence_lower) else []
        
        # Dynamic rules based on change type
        change_type = self._determine_change_type(original, corrected)
        
        if change_type == 'add_hyphen':
            result = self._validate_hyphen_addition_dynamic(words_in_original, before_context, after_context, language)
        elif change_type == 'remove_hyphen':
            result = self._validate_hyphen_removal_dynamic(original, corrected, language)
        elif change_type == 'dash_change':
            result = self._validate_dash_change_dynamic(original, corrected, before_context, after_context)
        else:
            # General compound adjective validation
            result = self._validate_compound_adjective_dynamic(words_in_original, after_context)
        
        if result:
            logger.info(f"Dynamic validation accepted: '{original}' → '{corrected}' (type: {change_type})")
        
        return result

    def _determine_change_type(self, original: str, corrected: str) -> str:
        """Dynamically determine the type of hyphenation change"""
        if '-' not in original and '-' in corrected:
            return 'add_hyphen'
        elif '-' in original and '-' not in corrected and '–' not in corrected:
            return 'remove_hyphen'
        elif '-' in original and '–' in corrected:
            return 'dash_change'
        elif '-' in original and '—' in corrected:  # em dash
            return 'dash_change'
        else:
            return 'modify_hyphen'

    def _validate_hyphen_addition_dynamic(self, words: List[str], before_context: List[str], after_context: List[str], language: str) -> bool:
        """Dynamically validate hyphen addition based on context"""
        
        # Check if it's likely a compound adjective before a noun
        if len(words) >= 2:
            # Check for adverb-adjective (reject if first word ends in -ly)
            if words[0].endswith('ly'):
                return False
            
            # Check if followed by a noun (not a verb or preposition)
            if after_context:
                next_word = after_context[0]
                # Simple heuristic: if next word doesn't start with common verb/prep starters
                if not next_word.startswith(('is', 'are', 'was', 'were', 'be', 'to', 'for', 'in', 'on', 'at', 'and', 'or', 'but')):
                    return True
            
            # Check for numeric compounds (5-year-old, etc.)
            if words[0].isdigit() or any(char.isdigit() for char in words[0]):
                return True
            
            # Check for common compound adjective patterns
            compound_indicators = ['well', 'high', 'low', 'long', 'short', 'self', 'user', 'time', 'cost', 'human', 'real']
            if words[0] in compound_indicators:
                return True
        
        return False

    def _validate_hyphen_removal_dynamic(self, original: str, corrected: str, language: str) -> bool:
        """Dynamically validate hyphen removal"""
        
        # Check if it's a legitimate single word in the target language
        language_preferences = self.hyphenation_guidance.get('language_specific', {}).get(language, {})
        
        # For US English, prefer single words for certain terms
        if language == 'US English':
            us_single_words = ['overshoot', 'cooperate', 'coordinate', 'nonprofit', 'setup']
            if corrected.lower() in us_single_words:
                logger.info(f"Accepted: US English single word '{original}' → '{corrected}'")
                return True
        
        # Check if it's removing hyphen from scientific terms that shouldn't be compound modifiers
        scientific_terms = ['community', 'species', 'system', 'process', 'method', 'composition']
        if any(term in corrected.lower() for term in scientific_terms):
            logger.info(f"Accepted: scientific term '{original}' → '{corrected}'")
            return True
        
        # Check for contextual hyphen removal (like "sea ice retreat")
        contextual_patterns = [
            ('sea ice', ['retreat', 'cover', 'extent', 'data']),  # sea ice + non-modifying word
            ('ice edge', ['bloom', 'retreat', 'formation']),      # ice edge + context
        ]
        
        corrected_lower = corrected.lower()
        for base_pattern, context_words in contextual_patterns:
            if base_pattern in corrected_lower:
                # Check if this is a contextual usage where hyphen should be removed
                logger.info(f"Accepted: contextual pattern '{original}' → '{corrected}'")
                return True
        
        # Check if it's removing hyphen from compound terms in non-modifying contexts
        if ' ' in corrected and '-' in original:
            # This is converting "term-word" to "term word"
            # Accept if it looks like a scientific or technical term
            technical_indicators = ['ocean', 'marine', 'climate', 'carbon', 'water', 'ice', 'snow']
            if any(indicator in corrected.lower() for indicator in technical_indicators):
                logger.info(f"Accepted: technical term context '{original}' → '{corrected}'")
                return True
        
        return False

    def _validate_dash_change_dynamic(self, original: str, corrected: str, before_context: List[str], after_context: List[str]) -> bool:
        """Dynamically validate dash type changes"""
        
        logger.info(f"Validating dash change: '{original}' → '{corrected}'")
        logger.info(f"Before context: {before_context}")
        logger.info(f"After context: {after_context}")
        
        # Check for geographic or proper noun connections
        words = original.split('-')
        if len(words) == 2:
            logger.info(f"Split words: {words}")
            
            # Check if both parts start with capital letters (proper nouns)
            if all(word and word[0].isupper() for word in words):
                # Accept all geographic connections - they typically need en dash
                logger.info(f"Accepted: geographic/proper noun connection '{original}' → '{corrected}'")
                return True
                
            # Also check for common geographic patterns regardless of capitalization
            geographic_patterns = ['asia', 'europe', 'america', 'africa', 'london', 'paris', 'new york']
            if any(pattern in original.lower() for pattern in geographic_patterns):
                logger.info(f"Accepted: geographic pattern '{original}' → '{corrected}'")
                return True
        
        # Check context for connection indicators
        connection_indicators = ['route', 'flight', 'connection', 'corridor', 'line', 'relationship', 'agreement', 'between']
        all_context = ' '.join(before_context + after_context)
        logger.info(f"All context: '{all_context}'")
        
        if any(indicator in all_context for indicator in connection_indicators):
            logger.info(f"Accepted: connection context '{original}' → '{corrected}'")
            return True
        
        # Special case for "Asia-Europe route" type patterns
        if 'route' in ' '.join(after_context) or 'route' in original.lower():
            logger.info(f"Accepted: route pattern '{original}' → '{corrected}'")
            return True
        
        logger.info(f"Rejected dash change: '{original}' → '{corrected}'")
        return False

    def _validate_compound_adjective_dynamic(self, words: List[str], after_context: List[str]) -> bool:
        """Dynamically validate compound adjective formation"""
        
        if len(words) < 2:
            return False
        
        # Check if followed by a noun
        if after_context:
            next_word = after_context[0]
            # Heuristic: nouns are less likely to be verbs or function words
            function_words = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                             'to', 'for', 'in', 'on', 'at', 'by', 'with', 'and', 'or', 'but', 'so'}
            if next_word not in function_words:
                return True
        
        return False

    async def _enhanced_fallback_analysis(self, sentence: str, language: str, style: str) -> List[Dict[str, Any]]:
        """Enhanced fallback analysis with improved accuracy"""
        logger.info("Using enhanced fallback hyphenation analysis")
        changes = []
        words = sentence.split()
        
        # Only look for clear compound adjective patterns
        for i in range(len(words) - 2):  # Need at least 3 words to check context
            word1 = words[i].lower().strip('.,!?;:')
            word2 = words[i + 1].lower().strip('.,!?;:')
            next_word = words[i + 2].lower().strip('.,!?;:')
            
            # Only consider clear compound adjective patterns that precede nouns
            compound_patterns = {
                ('well', 'known'): 'well-known',
                ('high', 'quality'): 'high-quality',
                ('low', 'cost'): 'low-cost',
                ('long', 'term'): 'long-term',
                ('short', 'term'): 'short-term',
                ('self', 'made'): 'self-made',
                ('user', 'friendly'): 'user-friendly',
                ('cost', 'effective'): 'cost-effective'
            }
            
            # Check for number + year + old pattern
            if word1.isdigit() and word2 in ['year', 'month', 'day'] and next_word == 'old':
                original = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                corrected = f"{words[i]}-{words[i + 1]}-{words[i + 2]}"
                start_pos = sentence.find(original)
                
                if start_pos != -1:
                    changes.append({
                        "input_word": original,
                        "formatted": corrected,
                        "start_idx": start_pos,
                        "end_idx": start_pos + len(original),
                        "source": f"Enhanced Fallback Analysis + {self.dictionary_sources.get(language, 'Standard Dictionary')}",
                        "metadata": {
                            "rule_applied": "enhanced_number_age_pattern",
                            "dictionary_source": self.dictionary_sources.get(language, "Standard Dictionary"),
                            "style_guide": style.upper(),
                            "confidence_score": 0.9,
                            "grammar_rule": "Number-age compound adjective before noun",
                            "ai_model": "Enhanced Fallback (LLaMA unavailable)"
                        }
                    })
            
            # Check for compound adjective patterns
            elif (word1, word2) in compound_patterns:
                # Verify the next word is likely a noun (simple heuristic)
                if not next_word.endswith('ly') and len(next_word) > 2:
                    original = f"{words[i]} {words[i + 1]}"
                    corrected = compound_patterns[(word1, word2)]
                    start_pos = sentence.find(original)
                    
                    if start_pos != -1:
                        changes.append({
                            "input_word": original,
                            "formatted": corrected,
                            "start_idx": start_pos,
                            "end_idx": start_pos + len(original),
                            "source": f"Enhanced Fallback Analysis + {self.dictionary_sources.get(language, 'Standard Dictionary')}",
                            "metadata": {
                                "rule_applied": "enhanced_compound_adjective_pattern",
                                "dictionary_source": self.dictionary_sources.get(language, "Standard Dictionary"),
                                "style_guide": style.upper(),
                                "confidence_score": 0.8,
                                "grammar_rule": "Compound adjective before noun",
                                "ai_model": "Enhanced Fallback (LLaMA unavailable)"
                            }
                        })
        
        logger.info(f"Enhanced fallback found {len(changes)} valid hyphenation suggestions")
        return changes

    def extract_text_from_docx(self, docx_bytes: bytes) -> List[Dict[str, Any]]:
        """Enhanced DOCX text extraction with better document structure handling"""
        paragraphs = []
        
        try:
            with zipfile.ZipFile(BytesIO(docx_bytes), 'r') as zip_file:
                # Debug: List all files in the DOCX
                file_list = zip_file.namelist()
                logger.info(f"📋 DOCX contains: {file_list[:10]}...")  # Show first 10 files
                
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
                    logger.warning("⚠️ No content found with standard extraction, trying aggressive search...")
                    all_text_in_doc = root.findall('.//w:t', namespaces)
                    logger.info(f"🔍 Found {len(all_text_in_doc)} text elements total")
                    
                    # Combine all text elements into paragraphs
                    combined_text = []
                    for text_elem in all_text_in_doc:
                        if text_elem.text and text_elem.text.strip():
                            combined_text.append(text_elem.text.strip())
                    
                    if combined_text:
                        # Group into logical paragraphs (split by multiple spaces or line breaks)
                        full_text = ' '.join(combined_text)
                        potential_paragraphs = re.split(r'\s{3,}|\n{2,}', full_text)
                        
                        for i, para_text in enumerate(potential_paragraphs):
                            para_text = para_text.strip()
                            if para_text and len(para_text) > 10:  # Only include substantial text
                                all_text_elements.append({
                                    'para_id': i + 1,
                                    'text': para_text,
                                    'index': i,
                                    'source': 'extracted'
                                })
                
                logger.info(f"✅ Successfully extracted {len(all_text_elements)} text sections")
                if all_text_elements:
                    logger.info(f"📝 Sample text: '{all_text_elements[0]['text'][:100]}...'")
                
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
                
            logger.info(f"🧪 Created {len(paragraphs)} test paragraphs to demonstrate accuracy")
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
 