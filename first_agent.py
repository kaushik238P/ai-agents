from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import WikipediaAPIWrapper

# ---- Step 1: Define tools ----
# @tool decorator makes a normal function usable by the agent
# Docstring is critical — agent reads it to decide when to use the tool

@tool
def calculator(expression: str) -> str:
    """Use this to calculate math expressions like 25 * 48 or 100 / 4."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def ai_knowledge(query: str) -> str:
    """Use this to answer questions about AI topics like RAG, Agents, LangChain, fine tuning, prompt engineering."""
    knowledge = {
        "rag": "RAG stands for Retrieval Augmented Generation. It retrieves relevant documents before generating answers.",
        "agents": "AI Agents follow the ReAct pattern - Reasoning and Acting in a loop until task is complete.",
        "langchain": "LangChain is a Python framework for building LLM applications with chains, memory, and agents.",
        "fine tuning": "Fine tuning trains an existing model on custom data. LoRA and QLoRA are popular techniques.",
        "prompt engineering": "Prompt engineering involves zero-shot, few-shot, and chain of thought techniques."
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return "I don't have specific knowledge about that topic."

@tool
def wikipedia_search(query: str) -> str:
    """Use this to search Wikipedia for facts about people, places, history, or general knowledge."""
    try:
        wiki = WikipediaAPIWrapper()
        result = wiki.run(query)
        return result[:500]
    except Exception as e:
        return f"Search error: {e}"

# ---- Step 2: Tools list ----
# Agent picks from these automatically based on the question
tools = [calculator, ai_knowledge, wikipedia_search]

# ---- Step 3: Create LLM ----
# ChatOllama supports tool calling — OllamaLLM does NOT
# This is the key difference — always use ChatOllama for agents
llm = ChatOllama(model="llama3.2", temperature=0)

# ---- Step 4: Create agent ----
# create_react_agent from langgraph sets up the full ReAct loop
# It connects llm + tools automatically
# No manual prompt needed — langgraph handles it internally
agent = create_react_agent(
    model=llm,
    tools=tools,
)

# ---- Step 5: Run agent ----
print("=" * 50)
print("AI AGENT — LANGGRAPH + CHATOOLLAMA")
print("=" * 50)

questions = [
    "What is 25 multiplied by 48?",
    "What is RAG in AI?",
    "Who invented Python programming language?",
]

for question in questions:
    print(f"\nQuestion: {question}")
    print("-" * 40)
    try:
        # messages format — same as what you learned in Month 1
        # HumanMessage wraps your question into the right format
        result = agent.invoke({
            "messages": [HumanMessage(content=question)]
        })

        # result["messages"] contains the full conversation
        # [-1] gets the last message which is always the final answer
        final = result["messages"][-1].content
        print(f"Answer: {final}")

    except Exception as e:
        print(f"Error: {e}")
    print("-" * 40)
