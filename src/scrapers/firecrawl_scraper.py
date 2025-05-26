"""
Firecrawl Scraper - Cloud-based scraping with advanced features
Enterprise-grade web scraping via Firecrawl API
"""

import aiohttp
import logging
from typing import Dict, Any, Optional, List
import asyncio

logger = logging.getLogger(__name__)

class FirecrawlScraper:
    """Firecrawl API-based scraper for advanced web scraping."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.firecrawl.dev/v0"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    async def scrape(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Scrape a URL using Firecrawl API.
        
        Args:
            url: Target URL
            **kwargs: Additional Firecrawl parameters
            
        Returns:
            Scraping result
        """
        endpoint = f"{self.base_url}/scrape"
        
        payload = {
            "url": url,
            "pageOptions": {
                "onlyMainContent": kwargs.get("only_main_content", True),
                "includeHtml": kwargs.get("include_html", True),
                "screenshot": kwargs.get("screenshot", False),
                "waitFor": kwargs.get("wait_for", 0)
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=self.headers
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if data.get("success"):
                        return {
                            "status": "success",
                            "url": url,
                            "content": data.get("data", {}).get("markdown", ""),
                            "html": data.get("data", {}).get("html", ""),
                            "metadata": data.get("data", {}).get("metadata", {}),
                            "screenshot": data.get("data", {}).get("screenshot"),
                            "extraction_method": "firecrawl"
                        }
                    else:
                        return {
                            "status": "error",
                            "error": data.get("error", "Unknown error"),
                            "url": url
                        }
                        
        except aiohttp.ClientResponseError as e:
            logger.error(f"Firecrawl API error: {e}")
            return {
                "status": "error",
                "error": f"API error: {e.status}",
                "url": url
            }
        except Exception as e:
            logger.error(f"Firecrawl scraping failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "url": url
            }
    
    async def crawl(
        self,
        url: str,
        max_pages: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Crawl a website using Firecrawl API.
        
        Args:
            url: Starting URL
            max_pages: Maximum pages to crawl
            **kwargs: Additional crawl parameters
            
        Returns:
            Crawl results
        """
        endpoint = f"{self.base_url}/crawl"
        
        payload = {
            "url": url,
            "crawlerOptions": {
                "limit": max_pages,
                "includes": kwargs.get("includes", []),
                "excludes": kwargs.get("excludes", []),
                "generateImgAltText": kwargs.get("generate_alt_text", False),
                "returnOnlyUrls": kwargs.get("only_urls", False),
                "maxDepth": kwargs.get("max_depth", 3)
            },
            "pageOptions": {
                "onlyMainContent": kwargs.get("only_main_content", True),
                "includeHtml": kwargs.get("include_html", False)
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Start crawl job
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=self.headers
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if not data.get("success"):
                        return {
                            "status": "error",
                            "error": data.get("error", "Failed to start crawl"),
                            "url": url
                        }
                    
                    job_id = data.get("jobId")
                    
                    # Poll for results
                    return await self._poll_crawl_job(job_id)
                    
        except Exception as e:
            logger.error(f"Firecrawl crawl failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "url": url
            }
    
    async def _poll_crawl_job(
        self,
        job_id: str,
        max_wait: int = 300,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """Poll for crawl job completion."""
        endpoint = f"{self.base_url}/crawl/status/{job_id}"
        
        elapsed = 0
        
        async with aiohttp.ClientSession() as session:
            while elapsed < max_wait:
                try:
                    async with session.get(
                        endpoint,
                        headers=self.headers
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                        
                        status = data.get("status")
                        
                        if status == "completed":
                            return {
                                "status": "success",
                                "pages": data.get("data", []),
                                "total_pages": len(data.get("data", [])),
                                "job_id": job_id
                            }
                        elif status == "failed":
                            return {
                                "status": "error",
                                "error": data.get("error", "Crawl job failed"),
                                "job_id": job_id
                            }
                        
                        # Still processing
                        await asyncio.sleep(poll_interval)
                        elapsed += poll_interval
                        
                except Exception as e:
                    logger.error(f"Error polling crawl job: {e}")
                    
        return {
            "status": "error",
            "error": "Crawl job timeout",
            "job_id": job_id
        }
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search using Firecrawl search API.
        
        Args:
            query: Search query
            limit: Number of results
            **kwargs: Additional search parameters
            
        Returns:
            Search results
        """
        endpoint = f"{self.base_url}/search"
        
        payload = {
            "query": query,
            "pageOptions": {
                "onlyMainContent": True,
                "fetchPageContent": kwargs.get("fetch_content", True),
                "includeHtml": False
            },
            "searchOptions": {
                "limit": limit
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=self.headers
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if data.get("success"):
                        return {
                            "status": "success",
                            "query": query,
                            "results": data.get("data", []),
                            "total_results": len(data.get("data", []))
                        }
                    else:
                        return {
                            "status": "error",
                            "error": data.get("error", "Search failed"),
                            "query": query
                        }
                        
        except Exception as e:
            logger.error(f"Firecrawl search failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }