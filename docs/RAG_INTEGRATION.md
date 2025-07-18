# Integration with RAG Frameworks

Using the [LettuceDetect Web API](API.md) it's easy to integrate halluciantion detection
into popular RAG frameworks like LangChain, LlamaIndex or Heystack.

## Demo Notebooks

The demo notebooks show how LettuceDetect can be used in applications using LangChain, LlamaIndex or Heystack. Check out the notebook for your library:

- `demo/langchain_demo.ipynb`
- `demo/llamaindex_demo.ipynb`
- `demo/haystack_demo.ipynb`

## "Chat with Website" Demo Application

A simple RAG demo application that lets you ask an LLM questions about a web page. It supports OpenAI and Ollama as LLM backends and LangChain, LlamaIndex and Heystack as RAG libraries. Look at the code to see how LettuceDetect could be integrated into you appliation.

Install dependencies:

```bash
pip install -r demo/streamlit_rag_demo/requirements.txt
```

Start LettuceDetect API (see [here](API.md) for more details):

```bash
python scripts/start_api.py dev
```

Run the demo:

```bash
streamlit run demo/streamlit_rag_demo/main.py
```
