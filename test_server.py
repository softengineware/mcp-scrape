#!/usr/bin/env python3
"""
Test script for MCP-Scrape server
Verifies all components are working correctly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main_v2 import mcp, scrape_context, startup, shutdown

async def test_server():
    """Run tests on the MCP server."""
    print("🧪 Testing MCP-Scrape Server")
    print("=" * 50)
    
    # Initialize server
    print("\n1️⃣ Testing server startup...")
    await startup()
    
    # Check server status
    print("\n2️⃣ Testing server status...")
    status = await mcp._tools["get_server_status"].function()
    print(f"Server status: {status['status']}")
    print(f"Available scrapers: {status['scrapers']}")
    print(f"Circuit breakers: {status.get('circuit_breakers', {})}")
    
    # Test HTTP scraping
    print("\n3️⃣ Testing HTTP scraping...")
    try:
        result = await mcp._tools["scrape_url"].function(
            url="https://example.com",
            extract_mode="http"
        )
        print(f"✅ HTTP scraping: {result['status']}")
        print(f"   Title: {result.get('metadata', {}).get('title', 'N/A')}")
    except Exception as e:
        print(f"❌ HTTP scraping failed: {e}")
    
    # Test Playwright if available
    if scrape_context and scrape_context.playwright_scraper:
        print("\n4️⃣ Testing Playwright scraping...")
        try:
            result = await mcp._tools["scrape_url"].function(
                url="https://example.com",
                extract_mode="playwright"
            )
            print(f"✅ Playwright scraping: {result['status']}")
        except Exception as e:
            print(f"❌ Playwright scraping failed: {e}")
    
    # Test search if Serper configured
    if scrape_context and scrape_context.serper_searcher:
        print("\n5️⃣ Testing web search...")
        try:
            result = await mcp._tools["search_web"].function(
                query="Python web scraping",
                num_results=3
            )
            print(f"✅ Web search: Found {result.get('total_results', 0)} results")
        except Exception as e:
            print(f"❌ Web search failed: {e}")
    
    # Cleanup
    print("\n6️⃣ Testing server shutdown...")
    await shutdown()
    print("✅ Server shutdown complete")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")

if __name__ == "__main__":
    # Set minimal environment for testing
    if not os.getenv("ENABLE_PLAYWRIGHT"):
        os.environ["ENABLE_PLAYWRIGHT"] = "false"  # Faster testing
    
    asyncio.run(test_server())