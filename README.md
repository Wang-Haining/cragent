# Cultural Revolution Archivist Agent

An AI-powered historian assistant specialized in the Chinese Cultural Revolution (1966-1976), 
designed to interface with the [University of Pittsburgh's Cultural Revolution Archive](https://culturalrevolution.pitt.edu/).

## 🚧 Prototype Status

**This is a prototype implementation.** The archive API integration is simulated and will need to be developed in collaboration with the Pittsburgh Cultural Revolution project team to:
- Define API endpoints for search, document retrieval, and analysis
- Establish authentication mechanisms
- Implement actual data access methods
- Create proper pagination and filtering capabilities

## Features

- 🔍 **Archive Search**: Natural language search across posters, documents, photographs, and newspapers
- 📄 **Document Analysis**: Upload and analyze historical documents with OCR support for Chinese text
- 🗓️ **Timeline Access**: Chronological exploration of Cultural Revolution events
- 🌐 **Bilingual Support**: Handles both English and Chinese (简体中文) content
- 🤖 **MCP Compatible**: Includes Model Context Protocol server for integration with other AI tools

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Tesseract OCR (for image text extraction)

### Installation


```bash

# clone the repository
git clone https://github.com/your-org/cr-archivist

cd cragemt


# install dependencies
pip install -r requirements.txt


# install Tesseract OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim

# macOS:
brew install tesseract tesseract-lang

```

### Running the Agent

```bash
# basic usage with default model
python agent.py

# specify a different model
python agent.py --model meta-llama/Meta-Llama-3.1-70B-Instruct

# custom port and options
python agent.py --port 8080 --temperature 0.8 --share
```

### Running the MCP Server (Optional)

```bash
# for integration with Claude Desktop or other MCP clients
python server.py
```

## Architecture

```
cr-archivist/
├── agent.py              # Main Gradio application
├── server.py             # MCP server implementation
├── system_prompt.md      # System prompt for the AI historian
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Usage Examples

### Ask Historical Questions
- "What was the May 16 Notification?"
- "Explain the role of Red Guards in the Cultural Revolution"
- "What happened during the January Storm in Shanghai?"

### Search the Archive
- "Search for big-character posters from 1967"
- "Find photographs of the Down to the Countryside Movement"
- "Show me documents about the Four Olds campaign"

### Analyze Documents
- Upload a PDF or image of a historical document
- Get translations, context, and historical analysis
- Understand propaganda techniques and symbolism

## Development Roadmap

### Phase 1: API Integration (Current Priority)
- [ ] Collaborate with Pittsburgh team on API specifications
- [ ] Implement real archive search endpoints
- [ ] Add document retrieval functionality
- [ ] Create metadata standardization

### Phase 2: Enhanced Features
- [ ] Advanced search filters (date ranges, locations, document types)
- [ ] Batch document processing
- [ ] Citation generation in academic formats
- [ ] Multi-document comparative analysis

### Phase 3: Research Tools
- [ ] Network analysis of historical figures
- [ ] Temporal pattern analysis
- [ ] Geographic mapping of events
- [ ] Integration with academic databases

[//]: # (## Technical Notes)

[//]: # ()
[//]: # (- **LLM**: Uses VLLM for efficient inference)

[//]: # (- **OCR**: Tesseract with Chinese language support)

[//]: # (- **Document Parsing**: Docling for PDF extraction)

[//]: # (- **UI**: Gradio for web interface)

[//]: # (- **Protocol**: MCP for tool standardization)

[//]: # (## Contributing)

[//]: # ()
[//]: # (This project is in active development. We welcome contributions, especially:)

[//]: # (- API design suggestions)

[//]: # (- Historical accuracy reviews)

[//]: # (- Chinese language support improvements)

[//]: # (- Tool functionality enhancements)

## Contact

For questions about the agent implementation, please open an issue in this repository.

[//]: # (For questions about the Cultural Revolution Archive, visit [culturalrevolution.pitt.edu]&#40;https://culturalrevolution.pitt.edu/&#41;.)

## License

MIT

---
