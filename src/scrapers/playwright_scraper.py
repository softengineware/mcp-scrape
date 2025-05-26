"""
Playwright Scraper - Browser automation for JavaScript-heavy sites
Military-grade browser automation with stealth capabilities
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from playwright.async_api import async_playwright, Page, Browser
import base64

from utils import (
    get_random_user_agent,
    should_rotate_user_agent,
    encode_image_to_base64
)

logger = logging.getLogger(__name__)

class PlaywrightScraper:
    """Advanced browser-based scraper using Playwright."""
    
    def __init__(self, browser: Browser):
        self.browser = browser
        self.stealth_enabled = True
        
    @classmethod
    async def create(cls, headless: bool = True):
        """Create a new PlaywrightScraper instance."""
        playwright = await async_playwright().start()
        
        # Launch browser with stealth options
        browser = await playwright.chromium.launch(
            headless=headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process'
            ]
        )
        
        return cls(browser)
    
    async def close(self):
        """Close the browser."""
        if self.browser:
            await self.browser.close()
    
    async def scrape(
        self,
        url: str,
        wait_for: Optional[str] = None,
        proxy: Optional[str] = None,
        user_agent: Optional[str] = None,
        viewport: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Scrape a URL using Playwright browser automation.
        
        Args:
            url: Target URL
            wait_for: CSS selector to wait for
            proxy: Proxy server URL
            user_agent: User agent string
            viewport: Browser viewport dimensions
            
        Returns:
            Scraping result
        """
        context_options = {}
        
        # Set user agent
        if not user_agent and should_rotate_user_agent():
            user_agent = get_random_user_agent()
        if user_agent:
            context_options['user_agent'] = user_agent
        
        # Set viewport
        if viewport:
            context_options['viewport'] = viewport
        else:
            context_options['viewport'] = {'width': 1920, 'height': 1080}
        
        # Set proxy
        if proxy:
            context_options['proxy'] = {'server': proxy}
        
        # Additional stealth settings
        context_options['bypass_csp'] = True
        context_options['ignore_https_errors'] = True
        
        try:
            # Create browser context
            context = await self.browser.new_context(**context_options)
            
            # Apply stealth scripts
            if self.stealth_enabled:
                await self._apply_stealth_scripts(context)
            
            # Create page
            page = await context.new_page()
            
            # Navigate to URL
            logger.info(f"🎭 Navigating to {url}")
            response = await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for specific element if requested
            if wait_for:
                logger.info(f"⏳ Waiting for selector: {wait_for}")
                await page.wait_for_selector(wait_for, timeout=10000)
            else:
                # Default wait for body to be loaded
                await page.wait_for_load_state('domcontentloaded')
            
            # Extract content
            content = await page.content()
            
            # Get page metadata
            metadata = await self._extract_page_metadata(page)
            
            # Extract links
            links = await self._extract_links(page)
            
            # Convert to markdown
            markdown = await self._content_to_markdown(page)
            
            # Take screenshot
            screenshot_bytes = await page.screenshot(full_page=False)
            screenshot_base64 = encode_image_to_base64(screenshot_bytes)
            
            # Clean up
            await context.close()
            
            return {
                "status": "success",
                "url": url,
                "content": markdown,
                "html": content,
                "metadata": metadata,
                "links": links,
                "screenshot": screenshot_base64,
                "extraction_method": "playwright",
                "status_code": response.status if response else None
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout scraping {url}")
            return {
                "status": "error",
                "error": "Page load timeout",
                "url": url
            }
        except Exception as e:
            logger.error(f"Playwright error for {url}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "url": url
            }
    
    async def screenshot(
        self,
        url: str,
        full_page: bool = True,
        wait_for: Optional[str] = None,
        viewport: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Take a screenshot of a webpage."""
        try:
            # Use scrape method but focus on screenshot
            result = await self.scrape(url, wait_for=wait_for, viewport=viewport)
            
            if result["status"] != "success":
                return result
            
            # Take full page screenshot if requested
            if full_page:
                context = await self.browser.new_context(
                    viewport=viewport or {'width': 1920, 'height': 1080}
                )
                page = await context.new_page()
                await page.goto(url, wait_until='networkidle')
                
                screenshot_bytes = await page.screenshot(full_page=True)
                screenshot_base64 = encode_image_to_base64(screenshot_bytes)
                
                await context.close()
                
                result["screenshot"] = screenshot_base64
                result["screenshot_type"] = "full_page"
            
            return {
                "status": "success",
                "url": url,
                "screenshot": result["screenshot"],
                "format": "png",
                "dimensions": viewport or {'width': 1920, 'height': 1080}
            }
            
        except Exception as e:
            logger.error(f"Screenshot failed for {url}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "url": url
            }
    
    async def interact(
        self,
        url: str,
        actions: List[Dict[str, Any]],
        extract_after: bool = True
    ) -> Dict[str, Any]:
        """
        Interact with a webpage through automated actions.
        
        Args:
            url: Target URL
            actions: List of actions to perform
            extract_after: Whether to extract content after interactions
            
        Returns:
            Interaction results
        """
        context = await self.browser.new_context()
        page = await context.new_page()
        
        try:
            # Navigate to URL
            await page.goto(url, wait_until='networkidle')
            
            action_results = []
            screenshots = []
            
            # Execute actions
            for action in actions:
                action_type = action.get('type')
                result = {"type": action_type, "status": "success"}
                
                try:
                    if action_type == 'click':
                        selector = action['selector']
                        await page.click(selector)
                        result["selector"] = selector
                        
                    elif action_type == 'fill':
                        selector = action['selector']
                        value = action['value']
                        await page.fill(selector, value)
                        result["selector"] = selector
                        result["value"] = value
                        
                    elif action_type == 'select':
                        selector = action['selector']
                        value = action['value']
                        await page.select_option(selector, value)
                        result["selector"] = selector
                        result["value"] = value
                        
                    elif action_type == 'wait':
                        time = action.get('time', 1000)
                        await page.wait_for_timeout(time)
                        result["time"] = time
                        
                    elif action_type == 'screenshot':
                        screenshot_bytes = await page.screenshot(full_page=False)
                        screenshot_base64 = encode_image_to_base64(screenshot_bytes)
                        screenshots.append({
                            "timestamp": asyncio.get_event_loop().time(),
                            "screenshot": screenshot_base64
                        })
                        result["screenshot_taken"] = True
                        
                    action_results.append(result)
                    
                except Exception as e:
                    result["status"] = "error"
                    result["error"] = str(e)
                    action_results.append(result)
                    logger.error(f"Action {action_type} failed: {e}")
            
            # Extract final content if requested
            final_content = None
            if extract_after:
                final_content = await self._content_to_markdown(page)
            
            # Get final URL (after any redirects)
            final_url = page.url
            
            await context.close()
            
            return {
                "status": "success",
                "url": url,
                "final_url": final_url,
                "action_results": action_results,
                "final_content": final_content,
                "screenshots": screenshots
            }
            
        except Exception as e:
            await context.close()
            logger.error(f"Interaction failed for {url}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "url": url
            }
    
    async def _apply_stealth_scripts(self, context):
        """Apply stealth scripts to avoid detection."""
        # Override navigator.webdriver
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        
        # Override other detection methods
        await context.add_init_script("""
            // Pass Chrome runtime check
            window.chrome = {
                runtime: {}
            };
            
            // Pass permissions check
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)
    
    async def _extract_page_metadata(self, page: Page) -> Dict[str, Any]:
        """Extract metadata from the page."""
        metadata = await page.evaluate("""
            () => {
                const getMeta = (name) => {
                    const element = document.querySelector(`meta[name="${name}"], meta[property="${name}"]`);
                    return element ? element.content : null;
                };
                
                return {
                    title: document.title,
                    description: getMeta('description') || getMeta('og:description'),
                    author: getMeta('author'),
                    keywords: getMeta('keywords'),
                    language: document.documentElement.lang,
                    canonical: document.querySelector('link[rel="canonical"]')?.href,
                    image: getMeta('og:image') || getMeta('twitter:image')
                };
            }
        """)
        
        metadata['url'] = page.url
        return metadata
    
    async def _extract_links(self, page: Page) -> List[Dict[str, str]]:
        """Extract all links from the page."""
        links = await page.evaluate("""
            () => {
                return Array.from(document.querySelectorAll('a[href]')).map(link => ({
                    url: link.href,
                    text: link.textContent.trim(),
                    title: link.title
                })).filter(link => link.url && !link.url.startsWith('javascript:'));
            }
        """)
        return links
    
    async def _content_to_markdown(self, page: Page) -> str:
        """Convert page content to markdown format."""
        # Simple markdown conversion
        # In production, you might want to use a more sophisticated approach
        text_content = await page.evaluate("""
            () => {
                // Remove scripts and styles
                const scripts = document.querySelectorAll('script, style');
                scripts.forEach(el => el.remove());
                
                // Get text content
                return document.body.innerText;
            }
        """)
        
        return text_content