# MCP-Scrape: Military-Grade Web Scraping MCP Server 🎯

A SEAL Team Six-grade web scraping MCP (Model Context Protocol) server that provides LLMs with powerful web scraping capabilities through multiple battle-tested strategies.

## 🚀 Key Improvements in v2

### 1. **FastMCP v2 Integration**
- Migrated to FastMCP v2 for better performance and reliability
- Proper startup/shutdown lifecycle management
- Global context management for scrapers

### 2. **Enhanced Error Handling**
- **Circuit Breaker Pattern**: Prevents cascading failures
- **Exponential Backoff with Jitter**: Smart retry strategies
- **Graceful Degradation**: Server continues even if some scrapers fail

### 3. **Vector Store Implementation**
- In-memory vector store for quick prototyping
- Supabase vector store support for production
- Document chunking and metadata management

### 4. **Improved Resilience**
- Rate limiting with configurable limits
- Concurrent request management
- Better timeout handling

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-scrape.git
cd mcp-scrape

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (if using Playwright)
python install_playwright.py
```

## 🛠️ Configuration

Create a `.env` file with your configuration:

```env
# Scraping Features
ENABLE_PLAYWRIGHT=true
ENABLE_VECTOR_STORE=false

# API Keys (optional)
FIRECRAWL_API_KEY=your_firecrawl_key
SERPER_API_KEY=your_serper_key
OPENAI_API_KEY=your_openai_key  # For LLM extraction
ANTHROPIC_API_KEY=your_anthropic_key  # Alternative LLM

# Rate Limiting
RATE_LIMIT_CPS=2  # Calls per second

# Proxy Configuration (optional)
PROXY_URL=http://proxy.example.com:8080
PROXY_ROTATION=true
USER_AGENT_ROTATION=true

# Vector Store (optional)
DATABASE_URL=your_supabase_url
```

## 🎮 Usage

### Running the Server

```bash
# Run with FastMCP v2
python src/main_v2.py

# Or run with MCP CLI
mcp run python src/main_v2.py
```

### Available Tools

#### 1. **scrape_url** - Single Page Scraping
```python
result = await scrape_url(
    url="https://example.com",
    extract_mode="auto",  # auto, playwright, firecrawl, http, readability
    wait_for="#content",  # CSS selector for dynamic content
    store_result=True     # Store in vector DB
)
```

#### 2. **crawl_site** - Multi-Page Crawling
```python
result = await crawl_site(
    start_url="https://example.com",
    max_depth=3,
    max_pages=100,
    url_pattern="/docs/*",  # Only crawl matching URLs
    concurrent_limit=5
)
```

#### 3. **search_web** - Web Search with Scraping
```python
result = await search_web(
    query="site:example.com AI tools",
    num_results=10,
    search_type="search",  # search, news, images, videos
    scrape_results=True    # Scrape top results
)
```

#### 4. **get_server_status** - Health Check
```python
status = await get_server_status()
# Returns scraper availability, circuit breaker states, etc.
```

## 🏗️ Architecture

### Scraping Strategies

1. **Playwright**: JavaScript-heavy sites, dynamic content
2. **Firecrawl**: Cloud-based scraping with anti-bot bypass
3. **HTTP**: Fast, lightweight for static content
4. **Readability**: Content extraction from cluttered pages

### Error Handling Features

#### Circuit Breaker States
- **CLOSED**: Normal operation
- **OPEN**: Too many failures, requests blocked
- **HALF_OPEN**: Testing recovery

#### Retry Strategy
```python
# Configurable exponential backoff
retry_config = {
    "max_retries": 5,
    "initial_delay": 1.0,
    "max_delay": 60.0,
    "backoff_factor": 2.0,
    "jitter": True  # Prevents thundering herd
}
```

## 🔧 Advanced Features

### Vector Store Integration
- Automatic document chunking
- Metadata preservation
- Similarity search support

### Rate Limiting
- Per-domain rate limiting
- Global rate limiting
- Automatic backoff on 429 errors

### Proxy Support
- HTTP/HTTPS proxy configuration
- Automatic proxy rotation
- Per-request proxy override

## 📊 Performance Tips

1. **Use Circuit Breakers**: Prevent cascading failures
2. **Enable Caching**: Reduce redundant requests
3. **Batch Operations**: Use `crawl_site` for multiple pages
4. **Monitor Status**: Check circuit breaker states regularly

## 🐛 Troubleshooting

### Common Issues

1. **Playwright not working**
   ```bash
   python install_playwright.py
   ```

2. **Rate limits hit**
   - Check circuit breaker status
   - Adjust RATE_LIMIT_CPS in .env

3. **Memory issues with vector store**
   - Use Supabase for production
   - Implement periodic cleanup

## 🚀 Future Improvements

- [ ] Redis-based distributed caching
- [ ] Kubernetes deployment support
- [ ] WebSocket support for real-time updates
- [ ] Built-in CAPTCHA solving
- [ ] Browser fingerprint rotation

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Pull requests welcome! Please follow the SEAL Team Six principles:
- Mission First
- No One Left Behind
- Continuous Improvement
- Adapt and Overcome