# MCP-Scrape: SEAL Team Six-Grade Web Scraping MCP Server

<p align="center">
  <img src="public/mcp-scrape-logo.png" alt="MCP-Scrape: Elite Web Scraping" width="600">
</p>

## 🎯 Mission Statement

MCP-Scrape is a military-grade web scraping MCP server that combines the most powerful scraping capabilities from multiple battle-tested tools. Built with SEAL Team Six principles: **Precision, Reliability, Adaptability, and No Failure**.

## 🚀 Core Capabilities

### **Operation Modes**
- **Stealth Mode**: Browser automation with anti-detection measures
- **Rapid Strike**: High-speed concurrent crawling with smart rate limiting
- **Deep Recon**: LLM-powered intelligent content extraction
- **Search & Destroy**: Advanced search with Google operators
- **Persistent Intel**: Vector storage for long-term memory

### **Weapon Systems**
1. **Playwright Engine**: Full browser automation with JavaScript rendering
2. **Firecrawl Integration**: Cloud-based scraping with advanced features
3. **Readability Extraction**: Clean content extraction from any webpage
4. **LLM Intelligence**: Natural language instructions for data extraction
5. **Proxy Arsenal**: Rotating proxies and user agents for evasion
6. **CAPTCHA Breaker**: Multiple strategies for anti-bot bypass

## 🛡️ Features

### **Core Scraping Tools**
- `scrape_url`: Single URL extraction with multiple fallback strategies
- `crawl_site`: Full site crawling with intelligent link following
- `search_web`: Advanced Google search with structured queries
- `extract_data`: LLM-powered data extraction with schemas
- `screenshot`: Visual capture of any webpage
- `interact`: Browser automation for dynamic content

### **Advanced Capabilities**
- **Multi-Strategy Approach**: Automatically tries different methods until success
- **Anti-Detection Suite**: User agent rotation, proxy support, SSL bypass
- **Rate Limiting**: Intelligent throttling to avoid blocks
- **Error Recovery**: Exponential backoff and retry logic
- **Content Cleaning**: Readability + custom extractors for clean data
- **Memory Integration**: Optional vector storage for RAG applications

## 📋 Prerequisites

- Python 3.12+
- Node.js 18+ (for JavaScript tools)
- Docker (recommended for deployment)
- API Keys (optional):
  - Firecrawl API key for cloud scraping
  - Serper API key for search functionality
  - OpenAI/Anthropic for LLM extraction
  - Proxy service credentials

## 🔧 Installation

### Quick Start with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-scrape.git
cd mcp-scrape

# Build the Docker image
docker build -t mcp/scrape .

# Run with environment variables
docker run --env-file .env -p 8080:8080 mcp/scrape
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-scrape.git
cd mcp-scrape

# Install Python dependencies
pip install -e .

# Install Node.js dependencies for JavaScript tools
cd js-tools && npm install && cd ..

# Install Playwright browsers
playwright install chromium

# Copy and configure environment
cp .env.example .env
# Edit .env with your configuration
```

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# Transport Configuration
TRANSPORT=sse              # or stdio
HOST=0.0.0.0
PORT=8080

# Scraping Engines
ENABLE_PLAYWRIGHT=true
ENABLE_FIRECRAWL=false    # Requires API key
ENABLE_SERPER=false       # Requires API key

# API Keys (optional)
FIRECRAWL_API_KEY=        # For Firecrawl cloud scraping
SERPER_API_KEY=           # For Google search
OPENAI_API_KEY=           # For LLM extraction

# Proxy Configuration (optional)
PROXY_URL=                # http://user:pass@proxy:port
PROXY_ROTATION=true
USER_AGENT_ROTATION=true

# Performance
MAX_CONCURRENT_REQUESTS=10
RATE_LIMIT_DELAY=1000     # milliseconds
TIMEOUT=30000             # milliseconds

# Storage (optional)
ENABLE_VECTOR_STORE=false
DATABASE_URL=             # PostgreSQL for vector storage
```

## 🚀 Running the Server

### SSE Transport (API Mode)

```bash
# Using Python
python src/main.py

# Using Docker
docker run --env-file .env -p 8080:8080 mcp/scrape
```

### Stdio Transport (Direct Integration)

Configure in your MCP client:

```json
{
  "mcpServers": {
    "scrape": {
      "command": "python",
      "args": ["/path/to/mcp-scrape/src/main.py"],
      "env": {
        "TRANSPORT": "stdio",
        "ENABLE_PLAYWRIGHT": "true"
      }
    }
  }
}
```

## 📚 Usage Examples

### Basic URL Scraping
```javascript
// Scrape a single URL with automatic strategy selection
await use_mcp_tool("scrape", "scrape_url", {
  url: "https://example.com",
  extract_mode: "auto"  // Tries all methods until success
});
```

### Advanced Extraction with LLM
```javascript
// Extract specific data using natural language
await use_mcp_tool("scrape", "extract_data", {
  url: "https://news.site.com",
  instruction: "Extract all article titles, authors, and publication dates",
  schema: {
    title: "string",
    author: "string",
    date: "date"
  }
});
```

### Full Site Crawling
```javascript
// Crawl entire website with filters
await use_mcp_tool("scrape", "crawl_site", {
  start_url: "https://docs.example.com",
  max_depth: 3,
  url_pattern: "/docs/*",
  concurrent_limit: 5
});
```

### Browser Interaction
```javascript
// Interact with dynamic content
await use_mcp_tool("scrape", "interact", {
  url: "https://app.example.com",
  actions: [
    { type: "click", selector: "#login-button" },
    { type: "fill", selector: "#username", value: "user" },
    { type: "wait", time: 2000 },
    { type: "screenshot", filename: "result.png" }
  ]
});
```

## 🏗️ Architecture

### Multi-Layer Scraping Strategy

```
┌─────────────────────────────────────────┐
│          MCP Client Request             │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│         Strategy Selector              │
│   (Chooses best approach for URL)     │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│          Scraping Engines              │
├─────────────────────────────────────────┤
│ 1. Playwright (JS rendering)           │
│ 2. Firecrawl (Cloud API)              │
│ 3. HTTP Client (Simple HTML)          │
│ 4. Readability (Content extraction)   │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│       Anti-Detection Layer             │
├─────────────────────────────────────────┤
│ • Proxy rotation                       │
│ • User agent spoofing                  │
│ • Rate limiting                        │
│ • CAPTCHA handling                     │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│        Content Processing              │
├─────────────────────────────────────────┤
│ • HTML cleaning                        │
│ • Markdown conversion                  │
│ • LLM extraction                       │
│ • Structured data parsing              │
└────────────────┬───────────────────────┘
                 │
┌────────────────▼───────────────────────┐
│         Optional Storage               │
│      (Vector DB for RAG)              │
└─────────────────────────────────────────┘
```

## 🎖️ SEAL Team Six Principles

This tool embodies elite military principles:

1. **Mission First**: Every scraping request completes successfully
2. **Failure is Not an Option**: Multiple fallback strategies ensure success
3. **Adapt and Overcome**: Automatic strategy selection based on target
4. **Leave No Trace**: Stealth mode with anti-detection measures
5. **Intelligence Driven**: LLM-powered smart extraction
6. **Team Coordination**: Modular architecture for easy extension

## 🔒 Security & Ethics

- Always respect robots.txt and website terms of service
- Use rate limiting to avoid overwhelming servers
- Only scrape publicly available information
- Implement proper authentication when required
- Store sensitive data securely

## 🤝 Contributing

We welcome contributions that enhance our scraping capabilities:

1. Fork the repository
2. Create a feature branch
3. Add your enhancement with tests
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

This project integrates the best features from:
- [Firecrawl](https://github.com/mendableai/firecrawl-mcp-server)
- [Crawl4AI](https://github.com/coleam00/mcp-crawl4ai-rag)
- [Playwright MCP](https://github.com/random-robbie/mcp-web-browser)
- [Serper MCP](https://github.com/marcopesani/mcp-server-serper)
- [Readability MCP](https://github.com/tolik-unicornrider/mcp_scraper)

---

**🚨 Remember: With great scraping power comes great responsibility. Use wisely.**