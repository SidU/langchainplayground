from langchain.agents import load_tools, initialize_agent, Tool, tool
from langchain.llms import OpenAI
from gpt_index import GPTSimpleVectorIndex, download_loader

@tool
def my_custom_tool(query: str) -> str:
    """Reverses the contents of the string."""
    # Reverse the string
    return query[::-1]

# Check if daIndx.json exists, and if so, load it.
# Otherwise, create a new index.
try:
    index = GPTSimpleVectorIndex.load_from_disk('daIndx.json')
except FileNotFoundError:
    RssReader = download_loader("RssReader")
    WikipediaReader = download_loader("WikipediaReader")
    loader = RssReader()
    documents = loader.load_data([
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://roelofjanelsinga.com/atom.xml"
    ])
    wikiLoader = WikipediaReader()
    wikiDocuments = wikiLoader.load_data(pages=['Berlin', 'Rome', 'Tokyo', 'Canberra', 'Santiago'])
    
    documents.extend(wikiDocuments)

    index = GPTSimpleVectorIndex(documents)
    index.save_to_disk('daIndx.json')

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Let's add a tool that uses the index we created above.
gptIndexTool = Tool(
        name = "GPT Index",
        func=lambda q: str(index.query(q)),
        description="useful for when you want to answer questions from New York Times or wikipedia pages of Berlin, Rome, Tokyo, Canberra and Santiago. The input to this tool should be a complete english sentence.",
        return_direct=True
    )

tools.append(gptIndexTool)

# Let's add a tool that reverses the input.
strReverseTool = Tool(
        name = "String Reverse Custom Tool",
        func=my_custom_tool,
        description="reverses the input string",
        return_direct=True
    )

tools.append(strReverseTool)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


while True:
    user_input = input("[Human ('e' to exit)]: ")
    if user_input == "e":
        print("[AI]: kthxbai")
        break
    agent.run(user_input)
