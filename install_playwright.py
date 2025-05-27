#!/usr/bin/env python3
"""
Playwright Installation Script
Installs Playwright browsers for MCP-Scrape
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_playwright():
    """Install Playwright and its browsers."""
    try:
        logger.info("🎭 Installing Playwright browsers...")
        
        # Check if playwright is installed
        try:
            import playwright
            logger.info("✅ Playwright package found")
        except ImportError:
            logger.error("❌ Playwright not installed. Please run: pip install playwright")
            return False
        
        # Install browsers
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Playwright Chromium browser installed successfully")
            logger.info(result.stdout)
            
            # Also install dependencies
            logger.info("📦 Installing system dependencies...")
            deps_result = subprocess.run(
                [sys.executable, "-m", "playwright", "install-deps", "chromium"],
                capture_output=True,
                text=True
            )
            
            if deps_result.returncode == 0:
                logger.info("✅ System dependencies installed")
            else:
                logger.warning(f"⚠️ System dependencies installation had issues: {deps_result.stderr}")
                logger.info("You may need to run with sudo: sudo python -m playwright install-deps chromium")
            
            return True
        else:
            logger.error(f"❌ Failed to install Playwright browsers: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during Playwright installation: {e}")
        return False

def verify_installation():
    """Verify Playwright installation."""
    try:
        from playwright.async_api import async_playwright
        import asyncio
        
        async def test_browser():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto("https://example.com")
                title = await page.title()
                await browser.close()
                return title == "Example Domain"
        
        result = asyncio.run(test_browser())
        if result:
            logger.info("✅ Playwright installation verified - browser works!")
            return True
        else:
            logger.error("❌ Playwright browser test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to verify Playwright: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 MCP-Scrape Playwright Setup")
    logger.info("=" * 50)
    
    # Check if running in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.info("✅ Running in virtual environment")
    else:
        logger.warning("⚠️ Not running in virtual environment - consider using one")
    
    # Install Playwright
    if install_playwright():
        logger.info("\n🔍 Verifying installation...")
        if verify_installation():
            logger.info("\n✅ Playwright setup complete! You can now use Playwright scraping.")
        else:
            logger.error("\n❌ Playwright verification failed. Please check the errors above.")
            sys.exit(1)
    else:
        logger.error("\n❌ Playwright installation failed. Please check the errors above.")
        sys.exit(1)