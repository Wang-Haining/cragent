"""MCP Server for Cultural Revolution Archivist

This implements a Model Context Protocol (MCP) server that exposes the Cultural Revolution
archive tools as MCP-compatible resources and tools.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import httpx
from urllib.parse import urljoin

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    ResourceContent,
    Tool,
    ToolArguments,
    ToolResult,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Archive configuration
ARCHIVE_BASE_URL = "https://culturalrevolution.pitt.edu"
ARCHIVE_API_BASE = f"{ARCHIVE_BASE_URL}/api"

class CulturalRevolutionMCPServer:
    """MCP Server for Cultural Revolution Archive"""

    def __init__(self):
        self.server = Server("cultural-revolution-archivist")
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP protocol handlers"""

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available archive resources"""
            return [
                Resource(
                    uri="cr://archive/search",
                    name="Archive Search",
                    description="Search the Cultural Revolution archive",
                    mime_type="application/json"
                ),
                Resource(
                    uri="cr://archive/timeline",
                    name="Historical Timeline",
                    description="Access Cultural Revolution timeline",
                    mime_type="application/json"
                ),
                Resource(
                    uri="cr://archive/collections",
                    name="Archive Collections",
                    description="Browse archive collections",
                    mime_type="application/json"
                ),
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> ResourceContent:
            """Read a specific resource"""
            if uri == "cr://archive/collections":
                # Return available collections
                collections = {
                    "collections": [
                        {
                            "id": "posters",
                            "name": "Political Posters",
                            "description": "Propaganda posters from the Cultural Revolution",
                            "count": 1200
                        },
                        {
                            "id": "documents",
                            "name": "Official Documents",
                            "description": "Government notices, directives, and reports",
                            "count": 850
                        },
                        {
                            "id": "photographs",
                            "name": "Historical Photographs",
                            "description": "Photos documenting the period",
                            "count": 3000
                        },
                        {
                            "id": "newspapers",
                            "name": "Newspapers and Periodicals",
                            "description": "Contemporary news sources",
                            "count": 500
                        }
                    ]
                }
                return ResourceContent(
                    uri=uri,
                    content=TextContent(
                        type="text",
                        text=json.dumps(collections, indent=2)
                    )
                )
            elif uri == "cr://archive/timeline":
                # Return timeline data
                timeline = {
                    "events": [
                        {
                            "date": "1966-05-16",
                            "title": "May 16 Notification",
                            "description": "CCP Central Committee issues circular starting the Cultural Revolution"
                        },
                        {
                            "date": "1966-08-08",
                            "title": "Sixteen Points",
                            "description": "Decision concerning the Great Proletarian Cultural Revolution"
                        },
                        {
                            "date": "1967-01",
                            "title": "January Storm",
                            "description": "Revolutionary committees seize power in Shanghai"
                        },
                        # Add more timeline events...
                    ]
                }
                return ResourceContent(
                    uri=uri,
                    content=TextContent(
                        type="text",
                        text=json.dumps(timeline, indent=2)
                    )
                )
            else:
                raise ValueError(f"Unknown resource: {uri}")

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="search_archive",
                    description="Search the Cultural Revolution archive for documents, images, and records",
                    arguments=ToolArguments(
                        type="object",
                        properties={
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "document_type": {
                                "type": "string",
                                "enum": ["all", "poster", "document", "photograph", "newspaper"],
                                "description": "Type of document to search for",
                                "default": "all"
                            },
                            "date_start": {
                                "type": "string",
                                "format": "date",
                                "description": "Start date (YYYY-MM-DD)"
                            },
                            "date_end": {
                                "type": "string",
                                "format": "date",
                                "description": "End date (YYYY-MM-DD)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results to return",
                                "default": 10
                            }
                        },
                        required=["query"]
                    )
                ),
                Tool(
                    name="get_document",
                    description="Retrieve and analyze a specific document from the archive",
                    arguments=ToolArguments(
                        type="object",
                        properties={
                            "document_id": {
                                "type": "string",
                                "description": "ID of the document to retrieve"
                            },
                            "include_translation": {
                                "type": "boolean",
                                "description": "Include English translation if available",
                                "default": True
                            }
                        },
                        required=["document_id"]
                    )
                ),
                Tool(
                    name="analyze_image",
                    description="Analyze a Cultural Revolution poster or photograph",
                    arguments=ToolArguments(
                        type="object",
                        properties={
                            "image_id": {
                                "type": "string",
                                "description": "ID of the image to analyze"
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": ["symbolic", "historical", "artistic", "full"],
                                "description": "Type of analysis to perform",
                                "default": "full"
                            }
                        },
                        required=["image_id"]
                    )
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> ToolResult:
            """Execute a tool"""

            if name == "search_archive":
                results = await self._search_archive(**arguments)
                return ToolResult(
                    content=TextContent(
                        type="text",
                        text=json.dumps(results, ensure_ascii=False, indent=2)
                    )
                )

            elif name == "get_document":
                document = await self._get_document(**arguments)
                if document.get("type") == "image":
                    return ToolResult(
                        content=ImageContent(
                            type="image",
                            image_url=document["url"],
                            alt_text=document.get("description", "Cultural Revolution document")
                        )
                    )
                else:
                    return ToolResult(
                        content=TextContent(
                            type="text",
                            text=json.dumps(document, ensure_ascii=False, indent=2)
                        )
                    )

            elif name == "analyze_image":
                analysis = await self._analyze_image(**arguments)
                return ToolResult(
                    content=TextContent(
                        type="text",
                        text=analysis
                    ),
                    embedded_resources=[
                        EmbeddedResource(
                            resource=Resource(
                                uri=f"cr://archive/images/{arguments['image_id']}",
                                name=f"Image {arguments['image_id']}",
                                mime_type="image/jpeg"
                            )
                        )
                    ]
                )

            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _search_archive(
            self,
            query: str,
            document_type: str = "all",
            date_start: Optional[str] = None,
            date_end: Optional[str] = None,
            limit: int = 10
    ) -> Dict[str, Any]:
        """Search the archive (simulated for demo)"""
        # In production, make actual API calls
        results = {
            "query": query,
            "total_results": 42,
            "results": [
                {
                    "id": "CR-1967-P-0234",
                    "type": "poster",
                    "title": "Destroy the Old World, Establish the New World",
                    "date": "1967-03-15",
                    "description": "Red Guard poster promoting the Four Olds campaign",
                    "thumbnail_url": f"{ARCHIVE_BASE_URL}/thumbnails/CR-1967-P-0234.jpg",
                    "url": f"{ARCHIVE_BASE_URL}/items/CR-1967-P-0234"
                },
                {
                    "id": "CR-1966-D-0089",
                    "type": "document",
                    "title": "Decision Concerning the Great Proletarian Cultural Revolution",
                    "date": "1966-08-08",
                    "description": "The Sixteen Points document outlining Cultural Revolution objectives",
                    "url": f"{ARCHIVE_BASE_URL}/items/CR-1966-D-0089"
                }
            ]
        }
        return results

    async def _get_document(
            self,
            document_id: str,
            include_translation: bool = True
    ) -> Dict[str, Any]:
        """Retrieve a specific document"""
        # Simulated document retrieval
        document = {
            "id": document_id,
            "type": "document",
            "title": "May 16 Notification",
            "date": "1966-05-16",
            "original_text": "中国共产党中央委员会通知...",
            "metadata": {
                "author": "CCP Central Committee",
                "location": "Beijing",
                "classification": "Official Document"
            }
        }

        if include_translation:
            document["translation"] = "Circular of the Central Committee of the Communist Party of China..."

        return document

    async def _analyze_image(
            self,
            image_id: str,
            analysis_type: str = "full"
    ) -> str:
        """Analyze a Cultural Revolution image"""
        # Simulated image analysis
        analysis = f"""
## Image Analysis: {image_id}

### Visual Elements
- Central figure: Revolutionary worker holding Little Red Book
- Background: Industrial machinery and red flags
- Color scheme: Predominantly red with gold accents
- Text: "革命无罪，造反有理" (Revolution is no crime, to rebel is justified)

### Historical Context
This poster exemplifies the visual propaganda style of the early Cultural Revolution period (1966-1968). 
The glorification of workers and peasants while holding Mao's quotations was a common theme.

### Symbolic Meaning
- Little Red Book: Represents Mao Zedong Thought as guiding ideology
- Industrial backdrop: Emphasizes continued production during revolution
- Upward gaze: Suggests optimism and forward movement

### Artistic Style
Classic Socialist Realist style with Chinese characteristics, featuring:
- Bold colors and strong contrasts
- Idealized human figures
- Dynamic composition suggesting movement and progress
"""
        return analysis

    async def run(self):
        """Run the MCP server"""
        async with stdio_server() as stream:
            await self.server.run(
                stream,
                options={
                    "name": "Cultural Revolution Archivist",
                    "version": "1.0.0"
                }
            )

# MCP client configuration for connecting to this server
MCP_CLIENT_CONFIG = {
    "mcpServers": {
        "cultural-revolution": {
            "command": "python",
            "args": ["mcp_server.py"],
            "env": {
                "ARCHIVE_API_KEY": "your-api-key-here"  # If needed
            }
        }
    }
}

if __name__ == "__main__":
    server = CulturalRevolutionMCPServer()
    asyncio.run(server.run())