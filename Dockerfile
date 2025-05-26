# Multi-stage build for optimal image size
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -e .

# Install Playwright browsers
RUN playwright install chromium
RUN playwright install-deps chromium

# Production stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # Required for Playwright
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libatspi2.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxcb1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    # Additional tools
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 scraper

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/ms-playwright /home/scraper/.cache/ms-playwright
COPY --from=builder /app /app

# Change ownership
RUN chown -R scraper:scraper /app /home/scraper

# Switch to non-root user
USER scraper

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PLAYWRIGHT_BROWSERS_PATH=/home/scraper/.cache/ms-playwright

# Default environment variables
ENV TRANSPORT=sse
ENV HOST=0.0.0.0
ENV PORT=8080
ENV ENABLE_PLAYWRIGHT=true
ENV ENABLE_VECTOR_STORE=false

# Expose port
EXPOSE 8080

# Health check (simple Python check since we don't have a health endpoint yet)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8080)); s.close()" || exit 1

# Run the MCP server
CMD ["python", "src/main.py"]