# AI Hyphenation Engine

An advanced AI-powered hyphenation checking system that uses LLaMA 3 (8B) and internet search to analyze documents for hyphenation issues based on comprehensive linguistic rules.

## Features

- **LLaMA 3 Integration**: Uses Meta's LLaMA 3 (8B) model for intelligent hyphenation analysis
- **Dictionary Validation**: Searches Merriam Webster (US English) or Oxford/Collins (UK English) dictionaries
- **Document Processing**: Handles DOCX files from URLs, breaking them into paragraphs and sentences
- **Rule-Based Analysis**: Follows comprehensive hyphenation rules for different English variants
- **Structured Output**: Returns detailed JSON with source sentences, changes, and metadata

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download NLTK Data** (automatically handled by the script):
```bash
python -c "import nltk; nltk.download('punkt')"
```

3. **GPU Setup** (Optional but recommended for LLaMA 3):
- Ensure CUDA is installed for GPU acceleration
- The engine will fall back to CPU if GPU is not available

## Usage

### Basic Usage

```python
import asyncio
from ai_hyphen_engine import HyphenationEngine

async def main():
    engine = HyphenationEngine()
    
    # Sample input
    input_data = {
        "url": "https://example.com/document.docx",
        "style": "apa",
        "check_list": "hyphen", 
        "eng": "US English"
    }
    
    results = await engine.process_document(input_data)
    print(engine.format_output(results))

asyncio.run(main())
```

### FastAPI Integration

```python
from ai_hyphen_engine import create_hyphenation_api
import uvicorn

app = create_hyphenation_api()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Testing

Run the test suite:
```bash
# Test full document processing
python test_hyphen_engine.py

# Test individual sentences only
python test_hyphen_engine.py --sentence
```

## Input Format

The engine expects input in this format:

```json
{
    "url": "https://example.com/document.docx",
    "style": "apa",
    "check_list": "hyphen",
    "eng": "US English"  // or "UK English"
}
```

## Output Format

Returns an array of objects, each containing:

```json
{
    "source_sentence": "The well known method is state of the art.",
    "changes": [
        {
            "input_word": "well known",
            "label": "HYPHEN_WORDS",
            "start_idx": 4,
            "end_idx": 14,
            "source": "Merriam Webster",
            "ocuring_count": 1,
            "formatted": "well-known",
            "type": "hyphenated",
            "unique_id": 1,
            "reason": "Compound adjective before noun"
        }
    ],
    "para_id": 1,
    "filename": "https://example.com/document.docx",
    "is_completed": false,
    "type": "hyphen"
}
```

## Hyphenation Rules

The engine follows comprehensive rules defined in the `rules` file:

1. **Dictionary Selection**: Merriam Webster for US English, Oxford/Collins for UK English
2. **Context Analysis**: Distinguishes compound adjectives vs. predicate adjectives
3. **Language Variants**: Handles US vs UK differences (e.g., "cooperation" vs "co-operation")
4. **Consistency Checking**: Flags inconsistent usage across documents
5. **Grammatical Rules**: Covers compound modifiers, adverb-adjective combinations, number-unit compounds

## Key Components

### HyphenationEngine Class

- `__init__()`: Initializes LLaMA 3 model and loads rules
- `process_document()`: Main function to process entire documents
- `process_sentence()`: Analyzes individual sentences
- `search_dictionary()`: Validates words against online dictionaries
- `analyze_sentence_with_llama()`: Uses LLM for intelligent analysis

### Fallback System

If LLaMA 3 fails to load, the engine falls back to:
1. Smaller transformer models
2. Rule-based pattern matching
3. Regex-based analysis

## Performance Considerations

- **GPU Recommended**: LLaMA 3 (8B) requires significant memory
- **8-bit Quantization**: Used for memory efficiency
- **Async Processing**: Supports concurrent document processing
- **Caching**: Dictionary lookups are optimized

## Error Handling

The engine includes robust error handling for:
- Network issues during document download
- Model loading failures
- Dictionary search timeouts
- Malformed documents

## Limitations

- Requires internet connection for dictionary validation
- LLaMA 3 model requires significant computational resources
- Processing time depends on document length and model performance

## Contributing

To extend the hyphenation rules:
1. Edit the `rules` file with new patterns
2. Test with `test_hyphen_engine.py`
3. Validate output format compliance

## License

This project is for educational and research purposes. 