"""
MCP-Scrape: SEAL Team Six-Grade Web Scraping MCP Server
Military-grade web scraping with multiple strategies and no failure option
"""

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
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
    store_to_vector_db
)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a dataclass for our application context
@dataclass
class ScrapeContext:
    """Context for the MCP Scrape server."""
    orchestrator: ScrapingOrchestrator
    playwright_scraper: Optional[PlaywrightScraper] = None
    firecrawl_scraper: Optional[FirecrawlScraper] = None
    http_scraper: HTTPScraper = None
    serper_searcher: Optional[SerperSearcher] = None
    vector_store: Optional[Any] = None

@asynccontextmanager
async def scrape_lifespan(server: FastMCP) -> AsyncIterator[ScrapeContext]:
    """
    Manages the scraping engines lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        ScrapeContext: The context containing all scraping engines
    """
    # Initialize scrapers based on configuration
    playwright_scraper = None
    firecrawl_scraper = None
    serper_searcher = None
    vector_store = None
    
    # Always initialize HTTP scraper as fallback
    http_scraper = HTTPScraper(
        user_agents=get_user_agents(),
        proxy_config=get_proxy_config()
    )
    
    # Initialize Playwright if enabled
    if os.getenv("ENABLE_PLAYWRIGHT", "true").lower() == "true":
        try:
            playwright_scraper = await PlaywrightScraper.create()
            logger.info("✅ Playwright scraper initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Playwright: {e}")
    
    # Initialize Firecrawl if API key provided
    if os.getenv("FIRECRAWL_API_KEY"):
        try:
            firecrawl_scraper = FirecrawlScraper(
                api_key=os.getenv("FIRECRAWL_API_KEY")
            )
            logger.info("✅ Firecrawl scraper initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Firecrawl: {e}")
    
    # Initialize Serper if API key provided
    if os.getenv("SERPER_API_KEY"):
        try:
            serper_searcher = SerperSearcher(
                api_key=os.getenv("SERPER_API_KEY")
            )
            logger.info("✅ Serper search initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Serper: {e}")
    
    # Initialize vector store if enabled
    if os.getenv("ENABLE_VECTOR_STORE", "false").lower() == "true":
        try:
            from vector_store import VectorStore
            vector_store = VectorStore(os.getenv("DATABASE_URL"))
            logger.info("✅ Vector store initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
    
    # Create the orchestrator
    orchestrator = ScrapingOrchestrator(
        playwright_scraper=playwright_scraper,
        firecrawl_scraper=firecrawl_scraper,
        http_scraper=http_scraper,
        serper_searcher=serper_searcher
    )
    
    try:
        yield ScrapeContext(
            orchestrator=orchestrator,
            playwright_scraper=playwright_scraper,
            firecrawl_scraper=firecrawl_scraper,
            http_scraper=http_scraper,
            serper_searcher=serper_searcher,
            vector_store=vector_store
        )
    finally:
        # Cleanup
        if playwright_scraper:
            await playwright_scraper.close()
        logger.info("🔒 Scraping engines shut down")

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-scrape",
    description="SEAL Team Six-grade web scraping MCP server with military precision",
    lifespan=scrape_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8080"))
)

@mcp.tool()
async def scrape_url(
    ctx: Context,
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
        ctx: MCP context containing scraping engines
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
    orchestrator = ctx.request_context.lifespan_context.orchestrator
    
    try:
        # Use orchestrator for intelligent scraping
        result = await orchestrator.scrape(
            url=url,
            strategy=extract_mode,
            wait_for=wait_for,
            proxy=proxy,
            user_agent=user_agent
        )
        
        # Store in vector database if requested
        if store_result and ctx.request_context.lifespan_context.vector_store:
            await store_to_vector_db(
                ctx.request_context.lifespan_context.vector_store,
                url,
                result["content"],
                result["metadata"]
            )
            result["stored"] = True
            
        return result
        
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "url": url
        }

@mcp.tool()
async def crawl_site(
    ctx: Context,
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
        ctx: MCP context containing scraping engines
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
    orchestrator = ctx.request_context.lifespan_context.orchestrator
    
    try:
        result = await orchestrator.crawl(
            start_url=start_url,
            max_depth=max_depth,
            max_pages=max_pages,
            url_pattern=url_pattern,
            concurrent_limit=concurrent_limit
        )
        
        # Store results if requested
        if store_results and ctx.request_context.lifespan_context.vector_store:
            for page in result["pages"]:
                await store_to_vector_db(
                    ctx.request_context.lifespan_context.vector_store,
                    page["url"],
                    page["content"],
                    page["metadata"]
                )
            result["stored"] = True
            
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
    ctx: Context,
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
        ctx: MCP context containing scraping engines
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
    if not ctx.request_context.lifespan_context.serper_searcher:
        return {
            "status": "error",
            "error": "Serper search not configured. Please provide SERPER_API_KEY."
        }
    
    try:
        # Perform search
        search_results = await ctx.request_context.lifespan_context.serper_searcher.search(
            query=query,
            num_results=num_results,
            search_type=search_type,
            location=location,
            time_range=time_range
        )
        
        # Optionally scrape the results
        if scrape_results and search_results.get("results"):
            orchestrator = ctx.request_context.lifespan_context.orchestrator
            scraped_content = []
            
            for result in search_results["results"][:5]:  # Limit scraping to top 5
                try:
                    scraped = await orchestrator.scrape(result["link"])
                    scraped_content.append({
                        "url": result["link"],
                        "title": result["title"],
                        "content": scraped["content"]
                    })
                except Exception as e:
                    logger.error(f"Failed to scrape {result['link']}: {e}")
                    
            search_results["scraped_content"] = scraped_content
            
        return search_results
        
    except Exception as e:
        logger.error(f"Search failed for '{query}': {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }

@mcp.tool()
async def extract_data(
    ctx: Context,
    url: str,
    instruction: str,
    schema: Optional[Dict[str, str]] = None,
    examples: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Extract structured data using LLM-powered intelligence.
    
    This tool combines web scraping with AI to extract specific information
    based on natural language instructions. Perfect for complex data extraction
    tasks that require understanding context.
    
    Args:
        ctx: MCP context containing scraping engines
        url: Target URL to extract data from
        instruction: Natural language description of what to extract
        schema: Optional schema defining the structure of extracted data
        examples: Optional examples to guide the extraction
        
    Returns:
        Dictionary containing:
        - extracted_data: Structured data matching the instruction
        - source_content: Original content that was analyzed
        - confidence: Confidence score of the extraction
    """
    orchestrator = ctx.request_context.lifespan_context.orchestrator
    
    try:
        # First scrape the content
        scraped = await orchestrator.scrape(url)
        
        if scraped["status"] != "success":
            return scraped
            
        # Use LLM to extract structured data
        extracted = await extract_with_llm(
            content=scraped["content"],
            instruction=instruction,
            schema=schema,
            examples=examples
        )
        
        return {
            "status": "success",
            "url": url,
            "extracted_data": extracted["data"],
            "confidence": extracted["confidence"],
            "source_content": scraped["content"][:500] + "..." if len(scraped["content"]) > 500 else scraped["content"]
        }
        
    except Exception as e:
        logger.error(f"Data extraction failed for {url}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "url": url
        }

@mcp.tool()
async def screenshot(
    ctx: Context,
    url: str,
    full_page: bool = True,
    wait_for: Optional[str] = None,
    viewport: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Capture a screenshot of a webpage.
    
    This tool provides visual intelligence by capturing webpage screenshots.
    Useful for verification, archival, or visual analysis.
    
    Args:
        ctx: MCP context containing scraping engines
        url: Target URL to screenshot
        full_page: Whether to capture the entire page or just viewport
        wait_for: CSS selector to wait for before screenshot
        viewport: Custom viewport dimensions {"width": 1920, "height": 1080}
        
    Returns:
        Dictionary containing:
        - screenshot: Base64 encoded image
        - format: Image format (png)
        - dimensions: Image dimensions
    """
    if not ctx.request_context.lifespan_context.playwright_scraper:
        return {
            "status": "error",
            "error": "Screenshot requires Playwright. Please enable ENABLE_PLAYWRIGHT."
        }
    
    try:
        result = await ctx.request_context.lifespan_context.playwright_scraper.screenshot(
            url=url,
            full_page=full_page,
            wait_for=wait_for,
            viewport=viewport
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Screenshot failed for {url}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "url": url
        }

@mcp.tool()
async def interact(
    ctx: Context,
    url: str,
    actions: List[Dict[str, Any]],
    extract_after: bool = True
) -> Dict[str, Any]:
    """
    Interact with dynamic web content through browser automation.
    
    This tool enables complex interactions with web applications,
    including form filling, clicking buttons, and navigating through
    multi-step processes.
    
    Args:
        ctx: MCP context containing scraping engines
        url: Target URL to interact with
        actions: List of actions to perform, each containing:
            - type: "click", "fill", "select", "wait", "screenshot"
            - selector: CSS selector for the element (not needed for wait/screenshot)
            - value: Value for fill/select actions
            - time: Wait time in milliseconds (for wait action)
            - filename: Filename for screenshot action
        extract_after: Whether to extract content after interactions
        
    Returns:
        Dictionary containing:
        - final_content: Page content after all interactions
        - action_results: Results of each action
        - final_url: URL after any redirects
        - screenshots: Any screenshots taken during interaction
    """
    if not ctx.request_context.lifespan_context.playwright_scraper:
        return {
            "status": "error",
            "error": "Browser interaction requires Playwright. Please enable ENABLE_PLAYWRIGHT."
        }
    
    try:
        result = await ctx.request_context.lifespan_context.playwright_scraper.interact(
            url=url,
            actions=actions,
            extract_after=extract_after
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Interaction failed for {url}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "url": url
        }

async def main():
    """Main entry point for the MCP server."""
    transport = os.getenv("TRANSPORT", "sse")
    
    logger.info(f"""
    🎯 MCP-SCRAPE: SEAL Team Six Web Scraping Server
    📡 Transport: {transport}
    🌐 Host: {os.getenv('HOST', '0.0.0.0')}
    🔌 Port: {os.getenv('PORT', '8080')}
    
    🛡️ Enabled Features:
    - Playwright: {os.getenv('ENABLE_PLAYWRIGHT', 'true')}
    - Firecrawl: {'✅' if os.getenv('FIRECRAWL_API_KEY') else '❌'}
    - Serper: {'✅' if os.getenv('SERPER_API_KEY') else '❌'}
    - Vector Store: {os.getenv('ENABLE_VECTOR_STORE', 'false')}
    
    🚀 Ready for deployment...
    """)
    
    if transport == 'sse':
        # Run the MCP server with SSE transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())