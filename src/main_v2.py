"""
MCP-Scrape: SEAL Team Six-Grade Web Scraping MCP Server
Military-grade web scraping with multiple strategies and no failure option
FastMCP v2 Implementation with enhanced error handling
"""

from fastmcp import FastMCP
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import json
import os
from typing import Optional, Dict, Any, List
import logging

# Import our scraping modules
from scrapers import (
    PlaywrightScraper,
    FirecrawlScraper,
    HTTPScraper,
    SerperSearcher,
    ScrapingOrchestrator
)
from utils import (
    get_user_agents,
    get_proxy_config,
    clean_content,
    extract_with_llm,
    store_to_vector_db,
    CircuitBreaker,
    retry_with_exponential_backoff
)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global context for scrapers
scrape_context = None

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-scrape",
    description="SEAL Team Six-grade web scraping MCP server with military precision"
)

@dataclass
class ScrapeContext:
    """Context for the MCP Scrape server."""
    orchestrator: ScrapingOrchestrator
    playwright_scraper: Optional[PlaywrightScraper] = None
    firecrawl_scraper: Optional[FirecrawlScraper] = None
    http_scraper: HTTPScraper = None
    serper_searcher: Optional[SerperSearcher] = None
    vector_store: Optional[Any] = None
    circuit_breakers: Dict[str, CircuitBreaker] = None

@mcp.server.startup()
async def startup():
    """Initialize all scraping engines on server startup."""
    global scrape_context
    
    logger.info("🚀 Starting MCP-Scrape server initialization...")
    
    # Initialize circuit breakers
    circuit_breakers = {
        "playwright": CircuitBreaker(failure_threshold=3, recovery_timeout=30),
        "firecrawl": CircuitBreaker(failure_threshold=5, recovery_timeout=60),
        "serper": CircuitBreaker(failure_threshold=5, recovery_timeout=45)
    }
    
    # Initialize scrapers with error handling
    playwright_scraper = None
    firecrawl_scraper = None
    serper_searcher = None
    vector_store = None
    
    # Always initialize HTTP scraper as fallback
    http_scraper = HTTPScraper(
        user_agents=get_user_agents(),
        proxy_config=get_proxy_config()
    )
    logger.info("✅ HTTP scraper initialized")
    
    # Initialize Playwright with retry
    if os.getenv("ENABLE_PLAYWRIGHT", "true").lower() == "true":
        try:
            async def init_playwright():
                return await PlaywrightScraper.create()
            
            playwright_scraper = await retry_with_exponential_backoff(
                init_playwright,
                max_retries=3,
                initial_delay=2.0
            )
            logger.info("✅ Playwright scraper initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Playwright after retries: {e}")
            logger.warning("Continuing without Playwright support")
    
    # Initialize Firecrawl if API key provided
    if os.getenv("FIRECRAWL_API_KEY"):
        try:
            firecrawl_scraper = FirecrawlScraper(
                api_key=os.getenv("FIRECRAWL_API_KEY")
            )
            logger.info("✅ Firecrawl scraper initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Firecrawl: {e}")
            logger.warning("Continuing without Firecrawl support")
    
    # Initialize Serper if API key provided
    if os.getenv("SERPER_API_KEY"):
        try:
            serper_searcher = SerperSearcher(
                api_key=os.getenv("SERPER_API_KEY")
            )
            logger.info("✅ Serper search initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Serper: {e}")
            logger.warning("Continuing without Serper search support")
    
    # Initialize vector store if enabled
    if os.getenv("ENABLE_VECTOR_STORE", "false").lower() == "true":
        try:
            from vector_store import VectorStore, SupabaseVectorStore
            
            # Use Supabase if URL provided, otherwise in-memory
            db_url = os.getenv("DATABASE_URL")
            if db_url and "supabase" in db_url:
                vector_store = SupabaseVectorStore(db_url)
            else:
                vector_store = VectorStore(db_url)
            
            await vector_store.connect()
            logger.info("✅ Vector store initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            logger.warning("Continuing without vector store support")
    
    # Create the orchestrator
    orchestrator = ScrapingOrchestrator(
        playwright_scraper=playwright_scraper,
        firecrawl_scraper=firecrawl_scraper,
        http_scraper=http_scraper,
        serper_searcher=serper_searcher
    )
    
    # Store context globally
    scrape_context = ScrapeContext(
        orchestrator=orchestrator,
        playwright_scraper=playwright_scraper,
        firecrawl_scraper=firecrawl_scraper,
        http_scraper=http_scraper,
        serper_searcher=serper_searcher,
        vector_store=vector_store,
        circuit_breakers=circuit_breakers
    )
    
    logger.info("✅ MCP-Scrape server initialization complete")

@mcp.server.shutdown()
async def shutdown():
    """Clean up resources on server shutdown."""
    global scrape_context
    
    if not scrape_context:
        return
    
    logger.info("🔒 Shutting down MCP-Scrape server...")
    
    try:
        # Close Playwright browser
        if scrape_context.playwright_scraper:
            await scrape_context.playwright_scraper.close()
            logger.info("✅ Playwright browser closed")
        
        # Disconnect vector store
        if scrape_context.vector_store:
            await scrape_context.vector_store.disconnect()
            logger.info("✅ Vector store disconnected")
        
        # Close HTTP session if needed
        if hasattr(scrape_context.http_scraper, 'session') and scrape_context.http_scraper.session:
            await scrape_context.http_scraper.session.close()
            logger.info("✅ HTTP session closed")
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    
    logger.info("🔒 MCP-Scrape server shutdown complete")

@mcp.tool()
async def scrape_url(
    url: str,
    extract_mode: str = "auto",
    wait_for: Optional[str] = None,
    proxy: Optional[str] = None,
    user_agent: Optional[str] = None,
    store_result: bool = False
) -> Dict[str, Any]:
    """
    Scrape a single URL using the best available strategy.
    
    This tool employs military-grade tactics to extract content from any webpage.
    It automatically selects the best scraping method and falls back to alternatives
    if the primary method fails.
    
    Args:
        url: Target URL to scrape
        extract_mode: Extraction strategy - "auto", "playwright", "firecrawl", "http", "readability"
        wait_for: CSS selector to wait for (Playwright only)
        proxy: Override default proxy settings
        user_agent: Override default user agent
        store_result: Whether to store in vector database
        
    Returns:
        Dictionary containing:
        - content: Extracted content (markdown format)
        - metadata: URL, title, extraction method, timestamp
        - screenshot: Base64 encoded screenshot (if available)
        - links: Extracted links from the page
        - status: Success/failure status
    """
    global scrape_context
    
    if not scrape_context:
        return {
            "status": "error",
            "error": "Server not initialized",
            "url": url
        }
    
    orchestrator = scrape_context.orchestrator
    
    try:
        # Use circuit breaker for the appropriate strategy
        if extract_mode == "playwright" and scrape_context.circuit_breakers["playwright"]:
            result = await scrape_context.circuit_breakers["playwright"].call(
                orchestrator.scrape,
                url=url,
                strategy=extract_mode,
                wait_for=wait_for,
                proxy=proxy,
                user_agent=user_agent
            )
        elif extract_mode == "firecrawl" and scrape_context.circuit_breakers["firecrawl"]:
            result = await scrape_context.circuit_breakers["firecrawl"].call(
                orchestrator.scrape,
                url=url,
                strategy=extract_mode,
                proxy=proxy,
                user_agent=user_agent
            )
        else:
            # Use orchestrator with retry for other strategies
            result = await retry_with_exponential_backoff(
                lambda: orchestrator.scrape(
                    url=url,
                    strategy=extract_mode,
                    wait_for=wait_for,
                    proxy=proxy,
                    user_agent=user_agent
                ),
                max_retries=3,
                initial_delay=1.0,
                max_delay=10.0
            )
        
        # Store in vector database if requested
        if store_result and scrape_context.vector_store:
            try:
                await store_to_vector_db(
                    scrape_context.vector_store,
                    url,
                    result["content"],
                    result["metadata"]
                )
                result["stored"] = True
            except Exception as e:
                logger.error(f"Failed to store in vector DB: {e}")
                result["stored"] = False
                result["storage_error"] = str(e)
        
        return result
        
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "url": url,
            "circuit_breaker_states": {
                name: cb.get_state() 
                for name, cb in scrape_context.circuit_breakers.items()
            } if scrape_context.circuit_breakers else {}
        }

@mcp.tool()
async def crawl_site(
    start_url: str,
    max_depth: int = 3,
    max_pages: int = 100,
    url_pattern: Optional[str] = None,
    concurrent_limit: int = 5,
    store_results: bool = False
) -> Dict[str, Any]:
    """
    Crawl an entire website with intelligent link following.
    
    This tool performs deep reconnaissance on a website, following links
    and extracting content from multiple pages. It respects rate limits
    and uses concurrent scraping for efficiency.
    
    Args:
        start_url: Starting URL for the crawl
        max_depth: Maximum link depth to follow
        max_pages: Maximum number of pages to crawl
        url_pattern: Regex pattern to filter URLs (e.g., "/docs/*")
        concurrent_limit: Maximum concurrent requests
        store_results: Whether to store in vector database
        
    Returns:
        Dictionary containing:
        - pages: List of scraped pages with content
        - total_pages: Total number of pages crawled
        - failed_urls: List of URLs that failed to scrape
        - sitemap: Discovered site structure
    """
    global scrape_context
    
    if not scrape_context:
        return {
            "status": "error",
            "error": "Server not initialized",
            "start_url": start_url
        }
    
    orchestrator = scrape_context.orchestrator
    
    try:
        result = await orchestrator.crawl(
            start_url=start_url,
            max_depth=max_depth,
            max_pages=max_pages,
            url_pattern=url_pattern,
            concurrent_limit=concurrent_limit
        )
        
        # Store results if requested
        if store_results and scrape_context.vector_store:
            stored_count = 0
            for page in result["pages"]:
                try:
                    await store_to_vector_db(
                        scrape_context.vector_store,
                        page["url"],
                        page["content"],
                        page["metadata"]
                    )
                    stored_count += 1
                except Exception as e:
                    logger.error(f"Failed to store page {page['url']}: {e}")
            
            result["stored"] = True
            result["stored_count"] = stored_count
        
        return result
        
    except Exception as e:
        logger.error(f"Crawl failed for {start_url}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "start_url": start_url
        }

@mcp.tool()
async def search_web(
    query: str,
    num_results: int = 10,
    search_type: str = "search",
    location: Optional[str] = None,
    time_range: Optional[str] = None,
    scrape_results: bool = False
) -> Dict[str, Any]:
    """
    Search the web using advanced Google search operators.
    
    This tool performs strategic intelligence gathering using web search.
    It can optionally scrape the content of search results for deeper analysis.
    
    Args:
        query: Search query (supports Google operators like site:, filetype:, etc.)
        num_results: Number of results to return
        search_type: Type of search - "search", "news", "images", "videos"
        location: Geographic location for localized results
        time_range: Time filter - "day", "week", "month", "year"
        scrape_results: Whether to scrape content from result URLs
        
    Returns:
        Dictionary containing:
        - results: List of search results with titles, URLs, snippets
        - scraped_content: Full content if scrape_results is True
        - total_results: Estimated total number of results
    """
    global scrape_context
    
    if not scrape_context or not scrape_context.serper_searcher:
        return {
            "status": "error",
            "error": "Serper search not configured. Please provide SERPER_API_KEY.",
            "query": query
        }
    
    try:
        # Use circuit breaker for Serper API
        search_results = await scrape_context.circuit_breakers["serper"].call(
            scrape_context.serper_searcher.search,
            query=query,
            num_results=num_results,
            search_type=search_type,
            location=location,
            time_range=time_range
        )
        
        # Optionally scrape the results
        if scrape_results and search_results.get("results"):
            orchestrator = scrape_context.orchestrator
            scraped_content = []
            
            # Limit concurrent scraping
            semaphore = asyncio.Semaphore(3)
            
            async def scrape_result(result):
                async with semaphore:
                    try:
                        scraped = await orchestrator.scrape(result["link"])
                        return {
                            "url": result["link"],
                            "title": result["title"],
                            "content": scraped["content"][:2000] + "..." if len(scraped["content"]) > 2000 else scraped["content"]
                        }
                    except Exception as e:
                        logger.error(f"Failed to scrape {result['link']}: {e}")
                        return None
            
            # Scrape top 5 results concurrently
            tasks = [scrape_result(result) for result in search_results["results"][:5]]
            scraped_results = await asyncio.gather(*tasks)
            
            # Filter out None results
            scraped_content = [r for r in scraped_results if r is not None]
            search_results["scraped_content"] = scraped_content
        
        return search_results
        
    except Exception as e:
        logger.error(f"Search failed for '{query}': {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query,
            "circuit_breaker_state": scrape_context.circuit_breakers["serper"].get_state()
        }

@mcp.tool()
async def get_server_status() -> Dict[str, Any]:
    """
    Get the current status of the MCP-Scrape server.
    
    Returns information about available scrapers, circuit breaker states,
    and overall server health.
    """
    global scrape_context
    
    if not scrape_context:
        return {
            "status": "error",
            "error": "Server not initialized"
        }
    
    status = {
        "status": "operational",
        "scrapers": {
            "playwright": scrape_context.playwright_scraper is not None,
            "firecrawl": scrape_context.firecrawl_scraper is not None,
            "http": scrape_context.http_scraper is not None,
            "serper": scrape_context.serper_searcher is not None
        },
        "features": {
            "vector_store": scrape_context.vector_store is not None,
            "circuit_breakers": scrape_context.circuit_breakers is not None
        }
    }
    
    # Add circuit breaker states
    if scrape_context.circuit_breakers:
        status["circuit_breakers"] = {
            name: {
                "state": cb.get_state(),
                "failure_count": cb.failure_count,
                "success_count": cb.success_count
            }
            for name, cb in scrape_context.circuit_breakers.items()
        }
    
    # Add vector store stats if available
    if scrape_context.vector_store:
        try:
            doc_count = await scrape_context.vector_store.count()
            status["vector_store_stats"] = {
                "document_count": doc_count,
                "connected": scrape_context.vector_store.is_connected
            }
        except Exception as e:
            status["vector_store_stats"] = {
                "error": str(e)
            }
    
    return status

# Main entry point for running the server
if __name__ == "__main__":
    logger.info(f"""
    🎯 MCP-SCRAPE: SEAL Team Six Web Scraping Server
    📡 FastMCP v2 Implementation
    
    🛡️ Configuration:
    - Playwright: {os.getenv('ENABLE_PLAYWRIGHT', 'true')}
    - Firecrawl: {'✅' if os.getenv('FIRECRAWL_API_KEY') else '❌'}
    - Serper: {'✅' if os.getenv('SERPER_API_KEY') else '❌'}
    - Vector Store: {os.getenv('ENABLE_VECTOR_STORE', 'false')}
    
    🚀 Ready for deployment...
    """)
    
    # Run the server
    mcp.run()