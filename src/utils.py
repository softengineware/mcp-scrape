"""
Utility functions for MCP-Scrape
Military-grade utilities for web scraping operations
"""

import os
import random
import re
import json
import base64
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
from urllib.parse import urlparse, urljoin
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

# User agents for rotation
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Firefox on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Safari on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    # Mobile Chrome
    "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    # Mobile Safari
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1"
]

def get_user_agents() -> List[str]:
    """Get list of user agents for rotation."""
    custom_agents = os.getenv("CUSTOM_USER_AGENTS", "").split(",")
    if custom_agents and custom_agents[0]:
        return custom_agents + USER_AGENTS
    return USER_AGENTS

def get_random_user_agent() -> str:
    """Get a random user agent."""
    return random.choice(get_user_agents())

def get_proxy_config() -> Optional[Dict[str, str]]:
    """Get proxy configuration from environment."""
    proxy_url = os.getenv("PROXY_URL")
    if not proxy_url:
        return None
    
    return {
        "http": proxy_url,
        "https": proxy_url
    }

def should_rotate_proxy() -> bool:
    """Check if proxy rotation is enabled."""
    return os.getenv("PROXY_ROTATION", "true").lower() == "true"

def should_rotate_user_agent() -> bool:
    """Check if user agent rotation is enabled."""
    return os.getenv("USER_AGENT_ROTATION", "true").lower() == "true"

def clean_content(html: str, preserve_links: bool = True) -> str:
    """
    Clean HTML content and extract text.
    
    Args:
        html: Raw HTML content
        preserve_links: Whether to preserve link information
        
    Returns:
        Cleaned text content
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Break into lines and remove leading/trailing space
    lines = (line.strip() for line in text.splitlines())
    
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    # Preserve links if requested
    if preserve_links:
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            text_content = link.get_text().strip()
            if text_content and href:
                links.append(f"[{text_content}]({href})")
        
        if links:
            text += "\n\n## Links\n" + "\n".join(links)
    
    return text

def extract_metadata(html: str, url: str) -> Dict[str, Any]:
    """
    Extract metadata from HTML.
    
    Args:
        html: Raw HTML content
        url: Page URL
        
    Returns:
        Dictionary containing metadata
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    metadata = {
        "url": url,
        "domain": urlparse(url).netloc,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    # Extract title
    title = soup.find('title')
    metadata["title"] = title.get_text().strip() if title else None
    
    # Extract meta tags
    for meta in soup.find_all('meta'):
        name = meta.get('name') or meta.get('property', '')
        content = meta.get('content', '')
        
        if name in ['description', 'og:description', 'twitter:description']:
            metadata["description"] = content
        elif name in ['author', 'article:author']:
            metadata["author"] = content
        elif name in ['keywords']:
            metadata["keywords"] = content.split(',')
        elif name in ['og:image', 'twitter:image']:
            metadata["image"] = content
    
    # Extract language
    html_tag = soup.find('html')
    if html_tag:
        metadata["language"] = html_tag.get('lang')
    
    return metadata

def extract_links(html: str, base_url: str) -> List[Dict[str, str]]:
    """
    Extract all links from HTML.
    
    Args:
        html: Raw HTML content
        base_url: Base URL for resolving relative links
        
    Returns:
        List of link dictionaries
    """
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    
    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text().strip()
        
        # Resolve relative URLs
        absolute_url = urljoin(base_url, href)
        
        # Skip certain types of links
        if absolute_url.startswith(('javascript:', 'mailto:', 'tel:', '#')):
            continue
            
        links.append({
            "url": absolute_url,
            "text": text,
            "relative": not href.startswith(('http://', 'https://'))
        })
    
    return links

def is_valid_url(url: str) -> bool:
    """Check if URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_domain(url: str) -> str:
    """Extract domain from URL."""
    return urlparse(url).netloc

def url_matches_pattern(url: str, pattern: str) -> bool:
    """Check if URL matches a pattern."""
    if not pattern:
        return True
    
    # Convert glob pattern to regex
    if '*' in pattern:
        pattern = pattern.replace('*', '.*')
    
    try:
        return bool(re.match(pattern, url))
    except:
        return False

async def retry_with_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions=(Exception,)
):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Result of the function
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= backoff_factor
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    raise last_exception

def calculate_content_hash(content: str) -> str:
    """Calculate hash of content for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()

def estimate_tokens(text: str) -> int:
    """Estimate token count for text."""
    # Rough estimation: 1 token ≈ 4 characters
    return len(text) // 4

def chunk_text(text: str, max_chunk_size: int = 2000) -> List[str]:
    """
    Split text into chunks for processing.
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph_size = len(paragraph)
        
        if current_size + paragraph_size > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(paragraph)
        current_size += paragraph_size
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

async def extract_with_llm(
    content: str,
    instruction: str,
    schema: Optional[Dict[str, str]] = None,
    examples: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Extract structured data from content using LLM.
    
    Supports OpenAI and Anthropic APIs based on available keys.
    
    Args:
        content: Content to analyze
        instruction: Natural language extraction instruction
        schema: Expected data schema
        examples: Example extractions
        
    Returns:
        Extracted data with confidence score
    """
    # Check for available LLM APIs
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        logger.warning("No LLM API keys configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        return {
            "data": {
                "notice": "LLM extraction requires API configuration",
                "instruction": instruction,
                "content_preview": content[:200] + "..."
            },
            "confidence": 0.0
        }
    
    # Build the prompt
    prompt = f"""Extract structured data from the following content based on the instruction.

Instruction: {instruction}
"""
    
    if schema:
        prompt += f"\n\nExpected Schema:\n{json.dumps(schema, indent=2)}"
    
    if examples:
        prompt += "\n\nExamples:"
        for i, example in enumerate(examples[:3]):
            prompt += f"\n\nExample {i+1}:\n{json.dumps(example, indent=2)}"
    
    prompt += f"\n\nContent to analyze:\n{content[:4000]}"  # Limit content length
    
    if len(content) > 4000:
        prompt += "\n\n[Content truncated...]"
    
    prompt += "\n\nReturn the extracted data as valid JSON."
    
    try:
        # Try OpenAI first
        if openai_key:
            try:
                from openai import AsyncOpenAI
                
                client = AsyncOpenAI(api_key=openai_key)
                response = await client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are a data extraction expert. Extract structured data from content and return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                extracted_data = json.loads(response.choices[0].message.content)
                
                return {
                    "data": extracted_data,
                    "confidence": 0.9,
                    "llm_provider": "openai",
                    "model": "gpt-4-turbo-preview"
                }
                
            except ImportError:
                logger.warning("OpenAI library not installed. Install with: pip install openai")
            except Exception as e:
                logger.error(f"OpenAI extraction failed: {e}")
        
        # Try Anthropic as fallback
        if anthropic_key:
            try:
                from anthropic import AsyncAnthropic
                
                client = AsyncAnthropic(api_key=anthropic_key)
                response = await client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=4000,
                    messages=[
                        {"role": "user", "content": prompt + "\n\nRespond only with valid JSON."}
                    ],
                    temperature=0.1
                )
                
                # Extract JSON from Claude's response
                response_text = response.content[0].text
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group())
                else:
                    extracted_data = {"error": "Could not parse JSON from response", "raw": response_text}
                
                return {
                    "data": extracted_data,
                    "confidence": 0.85,
                    "llm_provider": "anthropic",
                    "model": "claude-3-opus"
                }
                
            except ImportError:
                logger.warning("Anthropic library not installed. Install with: pip install anthropic")
            except Exception as e:
                logger.error(f"Anthropic extraction failed: {e}")
        
        # If all LLM attempts fail, try basic extraction
        return await extract_with_regex_fallback(content, instruction, schema)
        
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return {
            "data": {"error": str(e)},
            "confidence": 0.0
        }

async def extract_with_regex_fallback(
    content: str,
    instruction: str,
    schema: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Fallback extraction using regex patterns.
    """
    extracted = {}
    confidence = 0.3
    
    # Common extraction patterns
    patterns = {
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "phone": r'[\+]?[1-9][0-9 .\-\(\)]{8,}[0-9]',
        "url": r'https?://[^\s<>"{}|\\^\[\]`]+',
        "price": r'\$?\d+(?:,\d{3})*(?:\.\d{2})?',
        "date": r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}'
    }
    
    # Extract based on instruction keywords
    instruction_lower = instruction.lower()
    
    for pattern_name, pattern in patterns.items():
        if pattern_name in instruction_lower:
            matches = re.findall(pattern, content)
            if matches:
                extracted[pattern_name] = matches[0] if len(matches) == 1 else matches
                confidence = 0.5
    
    # Try to extract based on schema
    if schema and not extracted:
        for field, field_type in schema.items():
            if field_type == "string" and field in content:
                # Simple heuristic: look for field name followed by value
                pattern = f'{field}[:"]\s*([^"\n]+)'
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    extracted[field] = match.group(1).strip()
    
    return {
        "data": extracted,
        "confidence": confidence,
        "extraction_method": "regex_fallback"
    }

async def store_to_vector_db(
    vector_store: Any,
    url: str,
    content: str,
    metadata: Dict[str, Any]
) -> None:
    """
    Store content in vector database for RAG.
    
    Args:
        vector_store: Vector store instance
        url: Source URL
        content: Content to store
        metadata: Associated metadata
    """
    try:
        # Chunk content if necessary
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "url": url
            }
            
            await vector_store.add(
                content=chunk,
                metadata=chunk_metadata
            )
            
        logger.info(f"Stored {len(chunks)} chunks for {url}")
        
    except Exception as e:
        logger.error(f"Failed to store content in vector DB: {e}")
        raise

def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    return filename[:255]

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        async with self.lock:
            current_time = asyncio.get_event_loop().time()
            time_since_last_call = current_time - self.last_call
            
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                await asyncio.sleep(sleep_time)
            
            self.last_call = asyncio.get_event_loop().time()

def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

def decode_base64_to_image(base64_string: str) -> bytes:
    """Decode base64 string to image bytes."""
    return base64.b64decode(base64_string)

class CircuitBreaker:
    """
    Circuit breaker pattern for handling failures gracefully.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failure threshold exceeded, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == "OPEN":
                # Check if we should try half-open
                if self.last_failure_time:
                    time_since_failure = asyncio.get_event_loop().time() - self.last_failure_time
                    if time_since_failure >= self.recovery_timeout:
                        logger.info("Circuit breaker entering HALF_OPEN state")
                        self.state = "HALF_OPEN"
                        self.success_count = 0
                    else:
                        raise Exception(f"Circuit breaker OPEN, retry in {self.recovery_timeout - time_since_failure:.1f}s")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful execution."""
        async with self._lock:
            self.failure_count = 0
            
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    logger.info("Circuit breaker recovered to CLOSED state")
                    self.state = "CLOSED"
                    self.success_count = 0
    
    async def _on_failure(self):
        """Handle failed execution."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = asyncio.get_event_loop().time()
            
            if self.state == "HALF_OPEN":
                logger.warning("Circuit breaker failed in HALF_OPEN, returning to OPEN")
                self.state = "OPEN"
                self.success_count = 0
            elif self.failure_count >= self.failure_threshold:
                logger.error(f"Circuit breaker threshold exceeded ({self.failure_count} failures), entering OPEN state")
                self.state = "OPEN"
    
    def get_state(self) -> str:
        """Get current circuit breaker state."""
        return self.state

async def retry_with_exponential_backoff(
    func,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions=(Exception,)
):
    """
    Enhanced retry with exponential backoff and jitter.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for delay after each retry
        jitter: Add randomization to prevent thundering herd
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Result of the function
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            
            # Check if it's a rate limit error
            if "rate limit" in str(e).lower():
                logger.warning(f"Rate limit hit, using longer delay")
                delay = min(delay * 2, max_delay)
            
            if attempt < max_retries:
                # Add jitter to prevent thundering herd
                if jitter:
                    actual_delay = delay * (0.5 + random.random())
                else:
                    actual_delay = delay
                
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying in {actual_delay:.1f}s...")
                await asyncio.sleep(actual_delay)
                
                # Exponential backoff
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    raise last_exception