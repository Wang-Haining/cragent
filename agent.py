"""CRArchivist: A Cultural Revolution Historical Archive QA Agent

This Gradio application lets you interact with an AI historian specialized in the
Chinese Cultural Revolution, with the ability to search and analyze documents from
the University of Pittsburgh's Cultural Revolution Archive.

Workflow:

1. User asks a question about the Cultural Revolution.
2. The agent can:
   a. Answer from its knowledge base
   b. Search the Pittsburgh archive using MCP-compatible tools
   c. Analyze uploaded documents (PDFs, images, etc.)
3. Responses include proper citations and historical context.
"""

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import asyncio
from urllib.parse import urljoin, quote

import gradio as gr
import torch
import httpx
from docling.document_converter import DocumentConverter
from langchain_community.llms import VLLM
from transformers import AutoTokenizer
from PIL import Image
import pytesseract

# MCP Tool Protocol definitions
class MCPTool:
    """Base class for MCP-compatible tools"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.schema = self._define_schema()

    def _define_schema(self) -> dict:
        """Override in subclasses to define tool schema"""
        raise NotImplementedError

    async def execute(self, **kwargs) -> dict:
        """Override in subclasses to implement tool execution"""
        raise NotImplementedError

class ArchiveSearchTool(MCPTool):
    """Search the Cultural Revolution archive"""
    def __init__(self, base_url="https://culturalrevolution.pitt.edu"):
        super().__init__(
            name="search_archive",
            description="Search the University of Pittsburgh Cultural Revolution archive for documents, images, and records"
        )
        self.base_url = base_url
        self.search_endpoint = "/api/search"  # Adjust based on actual API

    def _define_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for the archive"
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "date_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string", "format": "date"},
                                "end": {"type": "string", "format": "date"}
                            }
                        },
                        "document_type": {
                            "type": "string",
                            "enum": ["poster", "document", "photograph", "newspaper", "leaflet", "all"]
                        },
                        "location": {"type": "string"},
                        "language": {
                            "type": "string",
                            "enum": ["chinese", "english", "both"]
                        }
                    }
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "maximum": 50
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str, filters: dict = None, limit: int = 10) -> dict:
        """Execute archive search"""
        # In production, this would make actual API calls to the archive
        # For now, we'll simulate the search
        async with httpx.AsyncClient() as client:
            try:
                # Construct search parameters
                params = {
                    "q": query,
                    "limit": limit
                }
                if filters:
                    params.update(filters)

                # Make request to archive API
                # response = await client.get(
                #     urljoin(self.base_url, self.search_endpoint),
                #     params=params
                # )

                # Simulated response for demonstration
                results = {
                    "status": "success",
                    "query": query,
                    "total_results": 42,
                    "results": [
                        {
                            "id": "CR-DOC-1966-001",
                            "title": "May 16 Notification",
                            "date": "1966-05-16",
                            "type": "document",
                            "description": "Central Committee circular initiating the Cultural Revolution",
                            "url": f"{self.base_url}/documents/may-16-notification",
                            "thumbnail": f"{self.base_url}/thumbnails/doc-1966-001.jpg"
                        },
                        # Add more simulated results as needed
                    ]
                }
                return results
            except Exception as e:
                return {"status": "error", "message": str(e)}

class DocumentAnalysisTool(MCPTool):
    """Analyze specific documents from the archive"""
    def __init__(self, base_url="https://culturalrevolution.pitt.edu"):
        super().__init__(
            name="analyze_document",
            description="Retrieve and analyze a specific document from the archive"
        )
        self.base_url = base_url

    def _define_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "ID of the document to analyze"
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["summary", "translation", "context", "full"],
                    "default": "summary"
                }
            },
            "required": ["document_id"]
        }

    async def execute(self, document_id: str, analysis_type: str = "summary") -> dict:
        """Retrieve and analyze a document"""
        # Simulated document retrieval and analysis
        return {
            "status": "success",
            "document_id": document_id,
            "analysis_type": analysis_type,
            "content": "Document content would be retrieved here",
            "metadata": {
                "date": "1966-08-08",
                "author": "Central Committee",
                "location": "Beijing"
            }
        }

class TimelineTool(MCPTool):
    """Get timeline of Cultural Revolution events"""
    def __init__(self):
        super().__init__(
            name="get_timeline",
            description="Retrieve chronological timeline of Cultural Revolution events"
        )

    def _define_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "event_type": {
                    "type": "string",
                    "enum": ["political", "social", "economic", "all"]
                }
            }
        }

    async def execute(self, start_date: str = None, end_date: str = None, event_type: str = "all") -> dict:
        """Get timeline of events"""
        # Would connect to timeline database
        return {
            "status": "success",
            "events": [
                {
                    "date": "1966-05-16",
                    "event": "May 16 Notification issued",
                    "significance": "Official start of the Cultural Revolution"
                },
                # More events...
            ]
        }

# Agent configuration
AGENT_NAME = "CRArchivist"
LLM = None
TOKENIZER = None
SYSTEM_PROMPT = None
MODEL_ID = None
MAX_CONVERSATION_ROUNDS = 50

# Initialize MCP tools
TOOLS = {
    "search_archive": ArchiveSearchTool(),
    "analyze_document": DocumentAnalysisTool(),
    "get_timeline": TimelineTool()
}

def _device() -> str:
    return (
        "cuda:1"
        if torch.cuda.device_count() > 1
        else "cuda:0" if torch.cuda.is_available() else "cpu"
    )

def load_llm_and_tokenizer(model_id: str, max_new: int = 2048, cli_args=None):
    """Loads the main VLLM instance and the tokenizer."""
    print("Loading VLLM and tokenizer...")
    is_70b = "70b" in model_id.lower()
    tensor_parallel_size = 2 if is_70b else 1
    enforce_eager = True if is_70b else False

    llm = VLLM(
        model=model_id,
        max_new_tokens=max_new,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        temperature=(
            cli_args.temperature
            if cli_args and hasattr(cli_args, "temperature")
            else 0.7
        ),
        top_p=cli_args.top_p if cli_args and hasattr(cli_args, "top_p") else 0.95,
    )
    print("VLLM loaded.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if tokenizer.chat_template:
        print(f"Tokenizer loaded with chat_template.")
    elif hasattr(tokenizer, "default_chat_template") and tokenizer.default_chat_template:
        print(f"Using default_chat_template.")
        tokenizer.chat_template = tokenizer.default_chat_template
    else:
        raise ValueError("No chat template available for tokenizer.")

    return llm, tokenizer

def parse_document(file_path: Path) -> str:
    """Parse various document types (PDF, images with OCR, etc.)"""
    file_ext = file_path.suffix.lower()

    if file_ext == '.pdf':
        try:
            text = DocumentConverter().convert(str(file_path)).document.export_to_text()
            return text
        except Exception as e:
            print(f"Error parsing PDF: {e}")
            raise gr.Error(f"Failed to parse PDF: {e}")

    elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
        try:
            # OCR for images (useful for historical documents)
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')  # Chinese + English
            return text
        except Exception as e:
            print(f"Error performing OCR: {e}")
            raise gr.Error(f"Failed to perform OCR: {e}")

    elif file_ext in ['.txt', '.md']:
        return file_path.read_text(encoding='utf-8')

    else:
        raise gr.Error(f"Unsupported file type: {file_ext}")

async def execute_tool_calls(tool_calls: List[Dict]) -> List[Dict]:
    """Execute MCP tool calls asynchronously"""
    results = []
    for call in tool_calls:
        tool_name = call.get("tool")
        params = call.get("parameters", {})

        if tool_name in TOOLS:
            tool = TOOLS[tool_name]
            try:
                result = await tool.execute(**params)
                results.append({
                    "tool": tool_name,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "error": str(e)
                })
        else:
            results.append({
                "tool": tool_name,
                "error": f"Unknown tool: {tool_name}"
            })

    return results

def format_llm_prompt_with_tools(
        system_prompt_str: str,
        document_content: Optional[str],
        tool_results: List[Dict],
        history_tuples: List[Tuple[str, str]],
        current_question_str: str,
        tokenizer_instance: Any,
) -> str:
    """Format prompt including tool results and document content"""
    messages = [{"role": "system", "content": system_prompt_str}]

    user_message_parts = []

    # Add document content if available
    if document_content:
        user_message_parts.append(
            f"**Uploaded Document Content:**\n{document_content}"
        )

    # Add tool results if available
    if tool_results:
        tool_results_str = json.dumps(tool_results, ensure_ascii=False, indent=2)
        user_message_parts.append(
            f"**Archive Search Results:**\n{tool_results_str}"
        )

    # Add conversation history
    if history_tuples:
        relevant_history = history_tuples[-MAX_CONVERSATION_ROUNDS:]
        history_parts = []
        for user_msg, assistant_msg in relevant_history:
            history_parts.append(f"User: {user_msg}")
            history_parts.append(f"Assistant: {assistant_msg}")

        history_str = "\n".join(history_parts)
        user_message_parts.append(
            f"**Conversation History:**\n{history_str}"
        )

    # Add current question
    user_message_parts.append(
        f"**Current Question:**\n{current_question_str}"
    )

    full_user_content = "\n\n---\n\n".join(user_message_parts)
    messages.append({"role": "user", "content": full_user_content})

    formatted_prompt = tokenizer_instance.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return formatted_prompt

def parse_tool_requests(response: str) -> List[Dict]:
    """Parse tool call requests from LLM response"""
    tool_calls = []

    # Simple pattern matching for tool calls
    # In production, use more robust parsing
    if "[SEARCH:" in response:
        # Extract search query
        start = response.find("[SEARCH:") + 8
        end = response.find("]", start)
        if end > start:
            query = response[start:end].strip()
            tool_calls.append({
                "tool": "search_archive",
                "parameters": {"query": query}
            })

    if "[ANALYZE:" in response:
        start = response.find("[ANALYZE:") + 9
        end = response.find("]", start)
        if end > start:
            doc_id = response[start:end].strip()
            tool_calls.append({
                "tool": "analyze_document",
                "parameters": {"document_id": doc_id}
            })

    return tool_calls

async def answer_with_tools(msg: str, state: Dict[str, Any]):
    """Answer questions using tools when needed"""
    if not msg.strip():
        yield state.get("ui_messages", []), state
        return

    document_text = state.get("document_text")
    ui_messages = state.get("ui_messages", []).copy()
    conversation_history = state.get("conversation_history", []).copy()

    ui_messages.append({"role": "user", "content": msg})
    ui_messages.append({"role": "assistant", "content": "🔍 Researching..."})
    yield ui_messages, state

    try:
        # First pass: Check if tools are needed
        initial_prompt = format_llm_prompt_with_tools(
            SYSTEM_PROMPT,
            document_text,
            [],
            conversation_history,
            msg,
            TOKENIZER
        )

        initial_response = LLM.invoke(initial_prompt)

        # Parse tool requests
        tool_calls = parse_tool_requests(initial_response)

        if tool_calls:
            # Execute tools
            ui_messages[-1]["content"] = "🔎 Searching the archive..."
            yield ui_messages, state

            tool_results = await execute_tool_calls(tool_calls)

            # Second pass with tool results
            final_prompt = format_llm_prompt_with_tools(
                SYSTEM_PROMPT,
                document_text,
                tool_results,
                conversation_history,
                msg,
                TOKENIZER
            )

            final_response = LLM.invoke(final_prompt)
            response_text = final_response.strip()
        else:
            response_text = initial_response.strip()

        if not response_text:
            response_text = "I apologize, but I couldn't generate a response. Please try rephrasing your question."

    except Exception as e:
        print(f"Error during answer generation: {e}")
        print(traceback.format_exc())
        response_text = "An error occurred while researching your question. Please try again."

    ui_messages[-1]["content"] = response_text
    state["conversation_history"].append((msg, response_text))
    state["ui_messages"] = ui_messages
    yield ui_messages, state

def upload_document(file_obj: gr.File, state: Dict[str, Any]):
    """Handle document upload"""
    if file_obj is None:
        raise gr.Error("Please upload a document.")

    try:
        text = parse_document(Path(file_obj.name))
        state["document_text"] = text

        ui_message = [
            {
                "role": "system",
                "content": f"Document uploaded successfully. You can now ask questions about its content or the Cultural Revolution archive."
            }
        ]
        state["ui_messages"] = ui_message

        return (
            ui_message,
            state,
            gr.update(interactive=True, placeholder="Ask about the document or search the archive...")
        )
    except Exception as e:
        raise gr.Error(f"Failed to process document: {e}")

def reset_session(state: Dict[str, Any]):
    """Reset the session"""
    initial_messages = [
        {
            "role": "system",
            "content": "Welcome! I'm your Cultural Revolution historian assistant. Ask me anything about the Cultural Revolution, upload documents for analysis, or I can search the Pittsburgh archive for you."
        }
    ]
    state["document_text"] = None
    state["conversation_history"] = []
    state["ui_messages"] = initial_messages
    return (
        initial_messages,
        state,
        gr.update(
            value="",
            placeholder="Ask about the Cultural Revolution or request archive searches...",
            interactive=True
        ),
    )

# Custom CSS for the interface
CUSTOM_CSS = """
body {font-family: 'Inter', sans-serif;}
#main-title-md h1 {text-align: center !important; font-size: 2.8em !important; margin-bottom: 0.1em !important; color: #8B0000 !important;}
#sub-title-md p {text-align: center !important; font-size: 1.3em !important; color: #555 !important; margin-top: 0 !important; margin-bottom: 2em !important;}
.gradio-chatbot > .wrap {display: flex; flex-direction: column;}
.gradio-chatbot .message-wrap[data-testid="user"] {align-self: flex-start !important;}
.gradio-chatbot .message-wrap[data-testid="user"] > div.message {background-color: #FFE4B5 !important; color: #000 !important; max-width: 70% !important; border-radius: 10px !important;}
.gradio-chatbot .message-wrap[data-testid="assistant"] {align-self: flex-end !important;}
.gradio-chatbot .message-wrap[data-testid="assistant"] > div.message {background-color: #F0E68C !important; color: #000 !important; max-width: 70% !important; border-radius: 10px !important;}
#footer-info-md p {font-size: 0.8em !important; color: #888 !important; text-align: center !important; margin-top: 25px !important; padding: 15px !important; border-top: 1px solid #eee !important;}
#footer-info-md a {color: #8B0000 !important; text-decoration: none !important;}
#footer-info-md a:hover {text-decoration: underline !important;}
"""

def build_ui(port: int, share_the_ui: bool):
    """Build the Gradio interface"""
    global MODEL_ID

    with gr.Blocks(css=CUSTOM_CSS, title="Cultural Revolution Archivist") as demo:
        gr.Markdown("# 📚 Cultural Revolution Archivist", elem_id="main-title-md")
        gr.Markdown("AI Historian for the Chinese Cultural Revolution Archive", elem_id="sub-title-md")

        initial_messages = [
            {
                "role": "system",
                "content": "Welcome! I'm your Cultural Revolution historian assistant. Ask me anything about the Cultural Revolution, upload documents for analysis, or I can search the Pittsburgh archive for you."
            }
        ]

        app_state = gr.State({
            "document_text": None,
            "conversation_history": [],
            "ui_messages": initial_messages,
        })

        with gr.Row():
            with gr.Column(scale=1):
                doc_file = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".txt", ".jpg", ".jpeg", ".png"]
                )
                reset_btn = gr.Button("🔄 Reset Session", variant="secondary")

                gr.Markdown("""
                ### Available Tools:
                - 🔍 **Archive Search**: Search the Cultural Revolution database
                - 📄 **Document Analysis**: Analyze specific documents
                - 📅 **Timeline**: Get chronological events

                ### Example Queries:
                - "Search for Red Guard posters from 1967"
                - "What was the May 16 Notification?"
                - "Analyze the Four Olds campaign"
                """)

            with gr.Column(scale=3):
                chat = gr.Chatbot(
                    value=initial_messages,
                    label="Chat with the Archivist",
                    height=500,
                    show_copy_button=True,
                    layout="panel",
                    type="messages",
                )

                msg_box = gr.Textbox(
                    lines=2,
                    placeholder="Ask about the Cultural Revolution or request archive searches...",
                    label="Your Question",
                    interactive=True,
                )

        # Event handlers
        doc_file.change(
            upload_document,
            inputs=[doc_file, app_state],
            outputs=[chat, app_state, msg_box],
        )

        reset_btn.click(
            reset_session,
            inputs=[app_state],
            outputs=[chat, app_state, msg_box]
        )

        # Use async wrapper for answer function
        def answer_wrapper(msg, state):
            return asyncio.run(answer_with_tools(msg, state))

        msg_box.submit(
            fn=answer_wrapper,
            inputs=[msg_box, app_state],
            outputs=[chat, app_state]
        )
        msg_box.submit(
            fn=lambda: "",
            inputs=None,
            outputs=[msg_box],
            queue=False
        )

        gr.Markdown(
            """
            <p style='text-align: center; color: #666; margin-top: 20px;'>
            Connected to the <a href='https://culturalrevolution.pitt.edu' target='_blank'>University of Pittsburgh Cultural Revolution Archive</a><br>
            Model: {model} | Device: {device}
            </p>
            """.format(model=MODEL_ID, device=_device()),
            elem_id="footer-info-md"
        )

    print(f"Starting Cultural Revolution Archivist on http://0.0.0.0:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=share_the_ui)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cultural Revolution Archivist: AI Historian with Archive Access"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Hugging Face model ID for the LLM.",
    )
    parser.add_argument(
        "--prompt",
        default="system_prompt.md",
        help="Path to the system prompt file.",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the app on."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for the LLM."
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Top_p for the LLM.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable external access via public Gradio link.",
    )
    args = parser.parse_args()

    MODEL_ID = args.model

    # Load system prompt
    system_prompt_path = Path(args.prompt)
    if not system_prompt_path.exists():
        # Create default historian prompt
        default_prompt = """You are CRArchivist, an AI historian specializing in the Chinese Cultural Revolution (1966-1976). You have access to the University of Pittsburgh's Cultural Revolution archive.

Your expertise includes:
- Deep knowledge of Cultural Revolution history, key figures, and events
- Understanding of political, social, and cultural impacts
- Ability to analyze primary sources and historical documents
- Access to search and analyze documents from the archive

When users ask questions:
1. Provide historically accurate information with proper context
2. Use archive search when specific documents or sources are needed (use [SEARCH: query] to search)
3. Analyze uploaded documents in historical context
4. Cite sources and provide document IDs when referencing archive materials
5. Explain complex historical events clearly
6. Maintain scholarly objectivity while being accessible

Always strive for historical accuracy and provide balanced perspectives on this complex period of Chinese history."""

        system_prompt_path.write_text(default_prompt)
        print(f"Created default historian prompt at '{system_prompt_path}'")

    SYSTEM_PROMPT = system_prompt_path.read_text().strip()
    print(f"Loaded system prompt from '{system_prompt_path}'")

    # Load LLM
    try:
        LLM, TOKENIZER = load_llm_and_tokenizer(
            MODEL_ID, max_new=args.max_new_tokens, cli_args=args
        )
    except Exception as e:
        print(f"Failed to load LLM: {e}")
        exit(1)

    # Start UI
    build_ui(args.port, share_the_ui=args.share)