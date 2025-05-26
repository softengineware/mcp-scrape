"""
Serper Search - Google search integration
Strategic intelligence gathering via search
"""

import aiohttp
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class SerperSearcher:
    """Serper API-based Google search integration."""
    
    def __init__(self, api_key: str, base_url: str = "https://google.serper.dev"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
    
    async def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "search",
        location: Optional[str] = None,
        time_range: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform a Google search using Serper API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            search_type: Type of search (search, news, images, videos)
            location: Geographic location
            time_range: Time filter (day, week, month, year)
            
        Returns:
            Search results
        """
        endpoint = f"{self.base_url}/{search_type}"
        
        payload = {
            "q": query,
            "num": num_results
        }
        
        # Add optional parameters
        if location:
            payload["location"] = location
            
        if time_range:
            payload["tbs"] = self._get_time_range_param(time_range)
        
        # Add any additional parameters
        payload.update(kwargs)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=self.headers
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Format results based on search type
                    if search_type == "search":
                        return self._format_search_results(data, query)
                    elif search_type == "news":
                        return self._format_news_results(data, query)
                    elif search_type == "images":
                        return self._format_image_results(data, query)
                    elif search_type == "videos":
                        return self._format_video_results(data, query)
                    else:
                        return {
                            "status": "success",
                            "query": query,
                            "results": data
                        }
                        
        except aiohttp.ClientResponseError as e:
            logger.error(f"Serper API error: {e}")
            return {
                "status": "error",
                "error": f"API error: {e.status}",
                "query": query
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
    
    def _get_time_range_param(self, time_range: str) -> str:
        """Convert time range to Google search parameter."""
        mapping = {
            "hour": "qdr:h",
            "day": "qdr:d",
            "week": "qdr:w",
            "month": "qdr:m",
            "year": "qdr:y"
        }
        return mapping.get(time_range, "")
    
    def _format_search_results(self, data: Dict, query: str) -> Dict[str, Any]:
        """Format standard search results."""
        organic = data.get("organic", [])
        
        results = []
        for item in organic:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "position": item.get("position"),
                "date": item.get("date"),
                "domain": item.get("domain")
            })
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "total_results": len(results),
            "search_information": data.get("searchInformation", {}),
            "knowledge_graph": data.get("knowledgeGraph"),
            "answer_box": data.get("answerBox"),
            "related_searches": data.get("relatedSearches", [])
        }
    
    def _format_news_results(self, data: Dict, query: str) -> Dict[str, Any]:
        """Format news search results."""
        news = data.get("news", [])
        
        results = []
        for item in news:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "date": item.get("date"),
                "source": item.get("source"),
                "imageUrl": item.get("imageUrl")
            })
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "total_results": len(results),
            "search_type": "news"
        }
    
    def _format_image_results(self, data: Dict, query: str) -> Dict[str, Any]:
        """Format image search results."""
        images = data.get("images", [])
        
        results = []
        for item in images:
            results.append({
                "title": item.get("title"),
                "imageUrl": item.get("imageUrl"),
                "link": item.get("link"),
                "source": item.get("source"),
                "thumbnailUrl": item.get("thumbnailUrl"),
                "imageWidth": item.get("imageWidth"),
                "imageHeight": item.get("imageHeight")
            })
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "total_results": len(results),
            "search_type": "images"
        }
    
    def _format_video_results(self, data: Dict, query: str) -> Dict[str, Any]:
        """Format video search results."""
        videos = data.get("videos", [])
        
        results = []
        for item in videos:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "date": item.get("date"),
                "source": item.get("source"),
                "channel": item.get("channel"),
                "duration": item.get("duration"),
                "thumbnailUrl": item.get("imageUrl")
            })
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "total_results": len(results),
            "search_type": "videos"
        }
    
    async def build_advanced_query(
        self,
        base_query: str,
        site: Optional[str] = None,
        filetype: Optional[str] = None,
        intitle: Optional[str] = None,
        inurl: Optional[str] = None,
        exclude: Optional[List[str]] = None,
        exact_phrase: Optional[str] = None,
        any_words: Optional[List[str]] = None
    ) -> str:
        """
        Build an advanced Google search query with operators.
        
        Args:
            base_query: Base search query
            site: Limit to specific site
            filetype: Limit to file type
            intitle: Must appear in title
            inurl: Must appear in URL
            exclude: Words to exclude
            exact_phrase: Exact phrase to match
            any_words: Any of these words
            
        Returns:
            Advanced search query
        """
        query_parts = [base_query]
        
        if site:
            query_parts.append(f"site:{site}")
            
        if filetype:
            query_parts.append(f"filetype:{filetype}")
            
        if intitle:
            query_parts.append(f"intitle:{intitle}")
            
        if inurl:
            query_parts.append(f"inurl:{inurl}")
            
        if exact_phrase:
            query_parts.append(f'"{exact_phrase}"')
            
        if any_words:
            query_parts.append(f"({' OR '.join(any_words)})")
            
        if exclude:
            for word in exclude:
                query_parts.append(f"-{word}")
        
        return " ".join(query_parts)