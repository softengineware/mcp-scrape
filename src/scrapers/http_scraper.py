"""
HTTP Scraper - Basic but reliable web scraping
Lightweight and fast for simple HTML pages
"""

import aiohttp
import asyncio
import logging
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
import html2text
import chardet

from ..utils import (
    get_random_user_agent,
    clean_content,
    extract_metadata,
    extract_links,
    should_rotate_user_agent
)

logger = logging.getLogger(__name__)

class HTTPScraper:
    """Simple HTTP-based scraper for static content."""
    
    def __init__(self, user_agents: List[str], proxy_config: Optional[Dict[str, str]] = None):
        self.user_agents = user_agents
        self.proxy_config = proxy_config
        self.session = None
        
        # Configure HTML to Markdown converter
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = False
        self.html2text.body_width = 0  # No line wrapping
        
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(ssl=False)  # Disable SSL verification for flexibility
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def scrape(
        self,
        url: str,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scrape a URL using simple HTTP request.
        
        Args:
            url: Target URL
            proxy: Override proxy configuration
            user_agent: Override user agent
            
        Returns:
            Scraping result
        """
        # Select user agent
        if not user_agent and should_rotate_user_agent():
            user_agent = get_random_user_agent()
        elif not user_agent:
            user_agent = self.user_agents[0]
        
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Use proxy if provided
        proxy_url = proxy or (self.proxy_config.get('http') if self.proxy_config else None)
        
        try:
            # Create session if not exists
            if not self.session:
                await self.__aenter__()
            
            logger.info(f"📡 HTTP GET: {url}")
            
            async with self.session.get(url, headers=headers, proxy=proxy_url) as response:
                # Check status
                response.raise_for_status()
                
                # Detect encoding
                content_bytes = await response.read()
                encoding = chardet.detect(content_bytes)['encoding'] or 'utf-8'
                html = content_bytes.decode(encoding, errors='ignore')
                
                # Extract content and metadata
                metadata = extract_metadata(html, url)
                links = extract_links(html, url)
                
                # Convert to markdown
                markdown_content = self.html2text.handle(html)
                
                # Clean content
                cleaned_content = clean_content(html)
                
                return {
                    "status": "success",
                    "url": url,
                    "content": markdown_content,
                    "cleaned_text": cleaned_content,
                    "html": html,
                    "metadata": metadata,
                    "links": links,
                    "extraction_method": "http",
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "encoding": encoding
                }
                
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error for {url}: {e}")
            return {
                "status": "error",
                "error": f"HTTP {e.status}: {e.message}",
                "url": url,
                "status_code": e.status
            }
        except asyncio.TimeoutError:
            logger.error(f"Timeout scraping {url}")
            return {
                "status": "error",
                "error": "Request timeout",
                "url": url
            }
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "url": url
            }
    
    async def extract_with_readability(self, html: str, url: str) -> Dict[str, Any]:
        """
        Apply Readability-style extraction to HTML.
        This is a simplified implementation - in production, 
        you might want to use python-readability or similar.
        
        Args:
            html: Raw HTML content
            url: Source URL
            
        Returns:
            Extraction result
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove clutter
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Find main content (simple heuristic)
            main_content = None
            
            # Try common content containers
            for selector in ['main', 'article', '[role="main"]', '#content', '.content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # Fallback to body
            if not main_content:
                main_content = soup.body or soup
            
            # Convert to markdown
            content_html = str(main_content)
            markdown_content = self.html2text.handle(content_html)
            
            # Extract metadata
            metadata = extract_metadata(html, url)
            
            return {
                "status": "success",
                "url": url,
                "content": markdown_content,
                "html": html,
                "metadata": metadata,
                "extraction_method": "readability"
            }
            
        except Exception as e:
            logger.error(f"Readability extraction failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "url": url
            }
    
    async def fetch_batch(
        self,
        urls: List[str],
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple URLs concurrently.
        
        Args:
            urls: List of URLs to fetch
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_one(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.scrape(url)
        
        tasks = [fetch_one(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)