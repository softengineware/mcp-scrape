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
    
    This is a placeholder for LLM integration.
    In production, this would call OpenAI, Anthropic, or other LLM APIs.
    
    Args:
        content: Content to analyze
        instruction: Natural language extraction instruction
        schema: Expected data schema
        examples: Example extractions
        
    Returns:
        Extracted data with confidence score
    """
    # Placeholder implementation
    # In production, integrate with actual LLM service
    
    logger.warning("LLM extraction not configured. Returning mock data.")
    
    return {
        "data": {
            "notice": "LLM extraction requires API configuration",
            "instruction": instruction,
            "content_preview": content[:200] + "..."
        },
        "confidence": 0.0
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