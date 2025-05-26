"""
Scraping Orchestrator - SEAL Team Six Command Center
Coordinates multiple scraping strategies with military precision
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Set
from urllib.parse import urlparse, urljoin
from collections import deque

from ..utils import (
    is_valid_url,
    url_matches_pattern,
    calculate_content_hash,
    retry_with_backoff,
    RateLimiter
)

logger = logging.getLogger(__name__)

class ScrapingOrchestrator:
    """
    Central command for coordinating scraping operations.
    Implements intelligent strategy selection and fallback mechanisms.
    """
    
    def __init__(
        self,
        playwright_scraper=None,
        firecrawl_scraper=None,
        http_scraper=None,
        serper_searcher=None
    ):
        self.playwright_scraper = playwright_scraper
        self.firecrawl_scraper = firecrawl_scraper
        self.http_scraper = http_scraper
        self.serper_searcher = serper_searcher
        
        # Rate limiter for crawling operations
        self.rate_limiter = RateLimiter(
            calls_per_second=float(os.getenv("RATE_LIMIT_CPS", "2"))
        )
        
        logger.info("🎯 Scraping Orchestrator initialized with available engines")
    
    async def scrape(
        self,
        url: str,
        strategy: str = "auto",
        wait_for: Optional[str] = None,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Scrape a URL using the specified or best available strategy.
        
        Args:
            url: Target URL
            strategy: Scraping strategy - auto, playwright, firecrawl, http, readability
            wait_for: CSS selector to wait for (Playwright only)
            proxy: Proxy to use
            user_agent: User agent to use
            
        Returns:
            Scraping result with content and metadata
        """
        if not is_valid_url(url):
            return {
                "status": "error",
                "error": f"Invalid URL: {url}",
                "url": url
            }
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Determine strategy
        if strategy == "auto":
            strategy = self._select_best_strategy(url)
            logger.info(f"Auto-selected strategy: {strategy} for {url}")
        
        # Try primary strategy
        try:
            if strategy == "playwright" and self.playwright_scraper:
                return await self._scrape_with_playwright(url, wait_for, proxy, user_agent)
            elif strategy == "firecrawl" and self.firecrawl_scraper:
                return await self._scrape_with_firecrawl(url)
            elif strategy == "http":
                return await self._scrape_with_http(url, proxy, user_agent)
            elif strategy == "readability":
                return await self._scrape_with_readability(url, proxy, user_agent)
            else:
                # Fallback to any available scraper
                return await self._scrape_with_fallback(url, proxy, user_agent)
                
        except Exception as e:
            logger.error(f"Primary strategy {strategy} failed: {e}")
            # Try fallback strategies
            return await self._scrape_with_fallback(url, proxy, user_agent, exclude=strategy)
    
    def _select_best_strategy(self, url: str) -> str:
        """Select the best scraping strategy based on URL and available engines."""
        domain = urlparse(url).netloc.lower()
        
        # Sites known to require JavaScript
        js_required_domains = [
            'twitter.com', 'x.com', 'instagram.com', 'facebook.com',
            'linkedin.com', 'tiktok.com', 'reddit.com', 'medium.com'
        ]
        
        # Check if JavaScript is likely required
        if any(domain.endswith(d) for d in js_required_domains):
            if self.playwright_scraper:
                return "playwright"
            elif self.firecrawl_scraper:
                return "firecrawl"
        
        # For static content, prefer faster methods
        if url.endswith(('.pdf', '.txt', '.csv', '.xml')):
            return "http"
        
        # Default strategy based on available engines
        if self.firecrawl_scraper:
            return "firecrawl"
        elif self.playwright_scraper:
            return "playwright"
        else:
            return "http"
    
    async def _scrape_with_playwright(
        self,
        url: str,
        wait_for: Optional[str] = None,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Scrape using Playwright browser automation."""
        logger.info(f"🎭 Scraping with Playwright: {url}")
        
        return await retry_with_backoff(
            lambda: self.playwright_scraper.scrape(
                url, wait_for=wait_for, proxy=proxy, user_agent=user_agent
            ),
            max_retries=2
        )
    
    async def _scrape_with_firecrawl(self, url: str) -> Dict[str, Any]:
        """Scrape using Firecrawl API."""
        logger.info(f"🔥 Scraping with Firecrawl: {url}")
        
        return await retry_with_backoff(
            lambda: self.firecrawl_scraper.scrape(url),
            max_retries=2
        )
    
    async def _scrape_with_http(
        self,
        url: str,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Scrape using simple HTTP client."""
        logger.info(f"🌐 Scraping with HTTP: {url}")
        
        return await retry_with_backoff(
            lambda: self.http_scraper.scrape(url, proxy=proxy, user_agent=user_agent),
            max_retries=3
        )
    
    async def _scrape_with_readability(
        self,
        url: str,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Scrape and extract content using Readability."""
        logger.info(f"📖 Scraping with Readability: {url}")
        
        # First get HTML with HTTP scraper
        result = await self._scrape_with_http(url, proxy, user_agent)
        
        if result["status"] == "success" and self.http_scraper:
            # Apply Readability extraction
            result = await self.http_scraper.extract_with_readability(
                result["html"], url
            )
        
        return result
    
    async def _scrape_with_fallback(
        self,
        url: str,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        exclude: Optional[str] = None
    ) -> Dict[str, Any]:
        """Try multiple scraping strategies until one succeeds."""
        strategies = []
        
        if self.playwright_scraper and exclude != "playwright":
            strategies.append(("playwright", self._scrape_with_playwright))
        if self.firecrawl_scraper and exclude != "firecrawl":
            strategies.append(("firecrawl", self._scrape_with_firecrawl))
        if exclude != "http":
            strategies.append(("http", self._scrape_with_http))
        
        last_error = None
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"Trying fallback strategy: {strategy_name}")
                
                if strategy_name == "playwright":
                    result = await strategy_func(url, proxy=proxy, user_agent=user_agent)
                elif strategy_name == "firecrawl":
                    result = await strategy_func(url)
                else:
                    result = await strategy_func(url, proxy, user_agent)
                
                if result["status"] == "success":
                    result["extraction_method"] = f"{strategy_name} (fallback)"
                    return result
                    
            except Exception as e:
                logger.error(f"Fallback strategy {strategy_name} failed: {e}")
                last_error = e
        
        # All strategies failed
        return {
            "status": "error",
            "error": f"All scraping strategies failed. Last error: {last_error}",
            "url": url
        }
    
    async def crawl(
        self,
        start_url: str,
        max_depth: int = 3,
        max_pages: int = 100,
        url_pattern: Optional[str] = None,
        concurrent_limit: int = 5
    ) -> Dict[str, Any]:
        """
        Crawl a website starting from the given URL.
        
        Args:
            start_url: Starting URL
            max_depth: Maximum link depth to follow
            max_pages: Maximum number of pages to crawl
            url_pattern: Pattern to filter URLs
            concurrent_limit: Maximum concurrent requests
            
        Returns:
            Crawl results with all scraped pages
        """
        logger.info(f"🕸️ Starting crawl from {start_url} (max_depth={max_depth}, max_pages={max_pages})")
        
        # Initialize crawl state
        visited: Set[str] = set()
        to_visit: deque = deque([(start_url, 0)])  # (url, depth)
        content_hashes: Set[str] = set()  # For deduplication
        pages = []
        failed_urls = []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def crawl_page(url: str, depth: int) -> Optional[Dict[str, Any]]:
            """Crawl a single page."""
            async with semaphore:
                if url in visited or len(pages) >= max_pages:
                    return None
                
                visited.add(url)
                
                try:
                    # Scrape the page
                    result = await self.scrape(url)
                    
                    if result["status"] != "success":
                        failed_urls.append({"url": url, "error": result.get("error")})
                        return None
                    
                    # Check for duplicate content
                    content_hash = calculate_content_hash(result["content"])
                    if content_hash in content_hashes:
                        logger.info(f"Skipping duplicate content: {url}")
                        return None
                    
                    content_hashes.add(content_hash)
                    
                    # Extract links for further crawling
                    if depth < max_depth and "links" in result:
                        for link in result["links"]:
                            link_url = link["url"]
                            
                            # Apply URL pattern filter
                            if url_pattern and not url_matches_pattern(link_url, url_pattern):
                                continue
                            
                            # Only crawl same domain
                            if urlparse(link_url).netloc == urlparse(start_url).netloc:
                                if link_url not in visited:
                                    to_visit.append((link_url, depth + 1))
                    
                    # Store page data
                    page_data = {
                        "url": url,
                        "depth": depth,
                        "content": result["content"],
                        "metadata": result.get("metadata", {}),
                        "links_count": len(result.get("links", []))
                    }
                    
                    pages.append(page_data)
                    logger.info(f"Crawled page {len(pages)}/{max_pages}: {url}")
                    
                    return page_data
                    
                except Exception as e:
                    logger.error(f"Failed to crawl {url}: {e}")
                    failed_urls.append({"url": url, "error": str(e)})
                    return None
        
        # Start crawling
        tasks = []
        
        while to_visit and len(pages) < max_pages:
            # Process batch of URLs
            batch = []
            while to_visit and len(batch) < concurrent_limit:
                url, depth = to_visit.popleft()
                if url not in visited:
                    batch.append((url, depth))
            
            # Crawl batch concurrently
            if batch:
                batch_tasks = [crawl_page(url, depth) for url, depth in batch]
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Crawl task failed: {result}")
        
        # Build sitemap
        sitemap = self._build_sitemap(pages, start_url)
        
        return {
            "status": "success",
            "start_url": start_url,
            "pages": pages,
            "total_pages": len(pages),
            "failed_urls": failed_urls,
            "sitemap": sitemap,
            "crawl_summary": {
                "max_depth_reached": max(p["depth"] for p in pages) if pages else 0,
                "unique_content": len(content_hashes),
                "duplicate_content": len(visited) - len(content_hashes)
            }
        }
    
    def _build_sitemap(self, pages: List[Dict[str, Any]], start_url: str) -> Dict[str, Any]:
        """Build a hierarchical sitemap from crawled pages."""
        sitemap = {
            "url": start_url,
            "children": {}
        }
        
        for page in pages:
            # Parse URL path
            parsed = urlparse(page["url"])
            path_parts = parsed.path.strip('/').split('/') if parsed.path else []
            
            # Build tree structure
            current = sitemap
            for part in path_parts:
                if part not in current["children"]:
                    current["children"][part] = {
                        "url": None,
                        "children": {}
                    }
                current = current["children"][part]
            
            current["url"] = page["url"]
            current["metadata"] = page["metadata"]
        
        return sitemap

# Add missing import
import os