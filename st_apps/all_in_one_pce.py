import os
import shutil
import base64
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from openai import OpenAI
from pdf2image import convert_from_path
import logging
from dataclasses import dataclass
from xml.etree import ElementTree as ET
import config

@dataclass
class ProcessingMetrics:
    pdf_to_png_time: float
    encoding_time: float
    llm_processing_time: float
    total_pages: int
    successful_pages: int
    failed_pages: int

class AllInOnePCE:
    def __init__(
        self,
        openai_api_key: str,
        tmp_pdf_png_path: str = config.PDF_PNG_PATH,
        model: str = config.LLM_MODEL,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
        top_p: float = config.LLM_TOP_P
    ):
        """Initialize the All In One PCE."""
        self.tmp_pdf_png_path = Path(tmp_pdf_png_path)
        self.tmp_pdf_png_path.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # Load prompts
        self.system_prompt, self.user_prompt = config.SYSTEM_PROMPT, config.USER_PROMPT


    def _convert_pdf_to_png(
        self, 
        pdf_path: str,
        dpi: int = 200,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None
    ) -> List[Path]:
        """Convert PDF pages to PNG images."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Create output subdirectory for this PDF
        pdf_name = pdf_path.stem
        tmp_pdf_png_subdir = self.tmp_pdf_png_path / pdf_name
        tmp_pdf_png_subdir.mkdir(exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=first_page,
            last_page=last_page
        )
        
        # Save images
        png_paths = []
        for i, image in enumerate(images, start=1):
            save_path = tmp_pdf_png_subdir / f"page_{i:03d}.png"
            image.save(save_path, "PNG")
            png_paths.append(save_path)
            print(f"Saved page {i} to {save_path}")
        
        return png_paths

    def _encode_images(self, png_paths: List[Path]) -> List[str]:
        """Encode PNG images to base64 strings."""
        base64_images = []
        for path in tqdm(png_paths, desc="Encoding images"):
            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                base64_images.append(encoded_string)
        return base64_images

    def _process_with_llm(self, base64_images: List[str]) -> List[Dict[str, Any]]:
        """Process images with OpenAI LLM."""
        responses = []
        for idx, base64_image in enumerate(tqdm(base64_images, desc="Processing with LLM")):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.user_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        },
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    response_format={"type": "text"},
                )
                print(f"Obtained response for page {idx + 1} from LLM: {self.model}")
                responses.append(response.choices[0].message.content)
            except Exception as e:
                print(f"Error processing page {idx + 1}: {e}")
                responses.append(None)
                
        return responses

    def _parse_xml_chunks(self, xml_string: str) -> Dict[str, Any]:
        """Parse XML response into structured data."""
        try:
            xml_string = xml_string.replace("```xml", "").replace("```", "")
            root = ET.fromstring(xml_string)
            chunks = {}
            
            for chunk in root.findall('./*'):
                chunk_data = {}
                for elem in chunk:
                    if elem.tag == 'metadata':
                        chunk_data['metadata'] = {
                            child.tag: child.text for child in elem
                        }
                    else:
                        chunk_data[elem.tag] = elem.text
                        
                chunks[chunk.tag] = chunk_data
                
            return chunks
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return {"error": str(e), "raw_content": xml_string}

    def process_pdf(
        self,
        pdf_path: str,
        save_output: bool = True,
        cleanup_images: bool = True
    ) -> tuple[List[Dict[str, Any]], ProcessingMetrics]:
        """Process a PDF file and extract semantic chunks."""
        start_time = time.time()
        
        # Process PDF
        png_start = time.time()
        png_paths = self._convert_pdf_to_png(pdf_path)
        png_time = time.time() - png_start
        
        encoding_start = time.time()
        base64_images = self._encode_images(png_paths)
        encoding_time = time.time() - encoding_start
        
        llm_start = time.time()
        llm_responses = self._process_with_llm(base64_images)
        llm_time = time.time() - llm_start
        
        # Parse chunks
        chunks = []
        successful_pages = 0
        failed_pages = 0
        
        for idx, response in enumerate(llm_responses):
            if response:
                try:
                    parsed_chunks = self._parse_xml_chunks(response)
                    chunks.extend(list(parsed_chunks.values()))
                    successful_pages += 1
                except Exception as e:
                    print(f"Failed to parse page {idx + 1}: {e}")
                    chunks.append({"error": str(e), "raw_content": response})
                    failed_pages += 1
            else:
                failed_pages += 1
                
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
                
        # Save results
        if save_output:
            output_path = Path(pdf_path).with_suffix('.json')
            with open(output_path, 'w') as f:
                json.dump(chunks, f, indent=2)
                
        # Cleanup
        if cleanup_images:
            try:
                shutil.rmtree(self.tmp_pdf_png_path)
                print(f"Cleaned up temporary directory: {self.tmp_pdf_png_path}")
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")
            
        # Return results
        metrics = ProcessingMetrics(
            pdf_to_png_time=png_time,
            encoding_time=encoding_time,
            llm_processing_time=llm_time,
            total_pages=len(png_paths),
            successful_pages=successful_pages,
            failed_pages=failed_pages
        )
        
        return chunks, metrics

if __name__ == "__main__":
    pass