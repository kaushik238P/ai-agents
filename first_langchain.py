from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---- Step 1: Create the LLM ----
llm = OllamaLLM(model="llama3.2")

# ---- Step 2: Create a prompt template ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI tutor for college students. Answer in 2-3 sentences only."),
    ("user", "Explain {topic} in simple words.")
])

# ---- Step 3: Create output parser ----
parser = StrOutputParser()

# ---- Step 4: Chain them together ----
chain = prompt | llm | parser

# ---- Step 5: Run the chain ----
print("=" * 50)
print("LANGCHAIN BASIC CHAIN")
print("=" * 50)

topics = ["RAG", "AI Agents", "LangChain"]

for topic in topics:
    print(f"\nTopic: {topic}")
    print("-" * 30)
    result = chain.invoke({"topic": topic})
    print(result)

# ---- Bonus: Change prompt without rewriting everything ----
print("\n" + "=" * 50)
print("DIFFERENT PROMPT, SAME CHAIN")
print("=" * 50)

interview_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a placement interviewer. Ask one interview question about the topic."),
    ("user", "Give me an interview question about {topic}.")
])

interview_chain = interview_prompt | llm | parser

for topic in ["RAG", "AI Agents"]:
    print(f"\nInterview Q for {topic}:")
    print(interview_chain.invoke({"topic": topic}))