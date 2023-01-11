from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


while True:
    user_input = input("[Human ('e' to exit)]: ")
    if user_input == "e":
        print("[AI]: kthxbai")
        break
    agent.run(user_input)
