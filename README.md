# Grid07 Cognitive Routing & RAG System

This repository contains a complete AI cognitive loop designed for the Grid07 platform. It implements vector-based persona matching, an autonomous content engine using LangGraph, and a robust RAG-based defense system against prompt injection.

## 🚀 Key Features

### Phase 1: Vector-Based Persona Matching
- Uses **ChromaDB** (in-memory) and **Sentence-Transformers** (`all-MiniLM-L6-v2`) to find the most relevant bot for any given post.
- Implements `route_post_to_bots` with similarity thresholding to ensure bots only engage with relevant content.

### Phase 2: Autonomous Content Engine (LangGraph)
- **Node-Based Orchestration**: A `StateGraph` manages the workflow from topic selection to final post generation.
- **Dynamic Research**: Bots use a `mock_searxng_search` tool to fetch real-world context before drafting opinions.
- **Strict JSON Output**: Guaranteed structured output using Pydantic and JSON mode.

### Phase 3: Combat Engine (RAG Defense)
- **Deep Thread understanding**: Bots analyze the entire `parent_post` and `comment_history`.
- **Injection Protection**: Implements a **Persona-First + XML Tagging** defense. Untrusted user input is wrapped in `<USER_INPUT>` tags, and the system is instructed to treat it as data, not instructions.

## 🛠 Tech Stack
- **Python 3.10+**
- **LangChain & LangGraph**
- **ChromaDB**
- **Sentence-Transformers**
- **Ollama / Groq / OpenAI** (Configurable)

## 📦 Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   - Copy `.env.example` to `.env`.
   - Set your preferred `LLM_PROVIDER` (default: `ollama`).
   - Add API keys if using `groq` or `openai`.

3. **Run the System**:
   ```bash
   python main.py
   ```

## 🛡 Prompt Injection Defense
The system uses a multi-layered defense strategy:
1. **Persona Anchoring**: Instructions are placed at the very beginning (Persona-First).
2. **Boundary Enforcement**: XML delimiters (`<THREAD_CONTEXT>`, `<USER_INPUT>`) prevent the LLM from confusing data with system directives.
3. **Explicit Denial**: The system prompt contains a "Critical Directive" that explicitly forbids any input inside tags from overriding the core mission.