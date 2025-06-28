import json
import asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from typing import List
from fastapi.responses import JSONResponse
from io import BytesIO
import requests
from docx import Document
from pydantic import BaseModel

# Import the processing function from our AI engine
from ai_hyphen_engine import process_sample_inputs

# Sample input data from Node backend
sample_input_us = {
    "url": "https://copyedit-dev.s3.us-east-2.amazonaws.com/1750772522543_QuoteshortDoc.docx",
    "style": "apa",
    "check_list": "hyphen",
    "eng": "US English"
}

sample_input_uk = {
    "url": "https://copyedit-dev.s3.us-east-2.amazonaws.com/1750772522543_QuoteshortDoc.docx",
    "style": "apa",
    "check_list": "hyphen",
    "eng": "UK English"
}

sample_input = {
    "url": "https://copyedit-dev.s3.us-east-2.amazonaws.com/1750772522543_QuoteshortDoc.docx",
    "style": "apa",
    "check_list": "hyphen",
    "eng": "US English"
}

print("Sample US input:", json.dumps(sample_input_us, indent=2))
print("Sample UK input:", json.dumps(sample_input_uk, indent=2))
print("real sample input:", json.dumps(sample_input, indent=2))

# Function to run hyphenation analysis
async def run_hyphenation_analysis():
    """Run hyphenation analysis on all sample inputs"""
    
    # Prepare sample inputs for processing
    samples = [
        {"name": "US English Sample", "input": sample_input_us},
        {"name": "UK English Sample", "input": sample_input_uk}, 
        {"name": "Default Sample", "input": sample_input}
    ]
    
    # Process all samples using the engine
    await process_sample_inputs(samples)

# Main execution
if __name__ == "__main__":
    print("\nStarting AI Hyphenation Processing...")
    asyncio.run(run_hyphenation_analysis())