#  AI Agents with LangGraph

ReAct-based AI agent built with LangGraph and Ollama — reasons and acts in a loop to answer questions using tools.

##  How it works
Question → Agent thinks → Picks a tool → Gets result → Thinks again → Final answer

##  Tools the agent uses
| Tool | Purpose |
|---|---|
| `calculator` | Solves math expressions |
| `ai_knowledge` | Answers AI concept questions |
| `wikipedia_search` | Searches Wikipedia for facts |

##  Tech Stack
- LangGraph (create_react_agent)
- ChatOllama (llama3.2)
- LangChain Tools

##  Run it
pip install langchain langchain-ollama langgraph wikipedia
ollama pull llama3.2
python first_agent.py

##  Key learning
- ChatOllama supports tool calling — OllamaLLM does NOT
- create_react_agent replaces deprecated AgentExecutor
- Agent automatically picks the right tool based on the question

##  Author
Kaushik — Mechanical Engineering @ SVNIT Surat, transitioning to AI Engineering