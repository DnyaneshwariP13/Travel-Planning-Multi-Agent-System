'''We need to use asynchronous programming (ASYNC/AWAIT) programming inside MCP because:
    - To prevent long running tasks from freezing the server and blocking the AI agent
    - As MCP servers handle input/output heavy operations, standard synchronous code would force the LLM to wait 
      completely idle for one task to finish before it could do anything else.
    - Using async allows the server to remain highly responsive and process multiple instructions concurrently. 
'''

import os
import asyncio
import certifi #to prevent path issues on windows
from dotenv import load_dotenv
load_dotenv()

from langchain_mcp_adapters.client import MultiServerMCPClient


#on windows os wemay get path issues with certifi, so we set the environment variable to the certifi path
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
AVIATIONSTACK_API_KEY=os.getenv("AVIATIONSTACK_API_KEY")


#creating client
client=MultiServerMCPClient(
    {
      "tavily":{
          "transport" : "streamable_http",
          "url" : f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
      },

      "Aviationstack": {
            "transport" : "stdio",
            "command": "uvx",
            "args": [
                "aviationstack-mcp"
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATIONSTACK_API_KEY 
            }
            },
    }  # type: ignore
)

#MCP client and MCP server are two most important part of MCP
#MCP client lives in our application
#MCP server is available on the internet by various providers


#this function all the tools available with MCP client
async def get_all_tools():
    #if we are async we need to use await
    tools=await client.get_tools()
    print("Available MCP tools:")
    for tool in tools:
        print(tool.name)

tavily_search_tool=None
#function returning only tavily search
async def get_tavily_search_tool():
    global tavily_search_tool
    if tavily_search_tool is not None:
        return tavily_search_tool

    tools = await client.get_tools()
    print("Available MCP tools:")
    for tool in tools:
        print(tool.name)

    for tool in tools:
        if tool.name == "tavily_search":
            tavily_search_tool = tool
            return tool

    raise RuntimeError("Tavily MCP tool 'tavily_search' not found.")



#--------------------------------Tavily and Aviation Tool--------------------------
search_tool= None
aviation_tools={}

async def initialize_mcp():

    global search_tool
    global aviation_tools

    if search_tool is not None and aviation_tools:
        return


    tools=await client.get_tools()

    print("\n Available MCP Tools:\n")

    for tool in tools:
        print(tool.name)

    search_tool=next(
        tool
        for tool in tools if tool.name=="tavily_search"
    )

    aviation_tools={
        tool.name: tool
        for tool in tools if tool.name!="tavily_search"
    }

async def tavily_mcp_search(query: str):
    await initialize_mcp()
    if search_tool is None:
        raise RuntimeError("Initialize MCP search tool could not be initialized.")

    result = await search_tool.ainvoke(
        {
            "query": query
        }
    )
    #print(result)
    return result


async def aviation_mcp_call(tool_name:str, tool_args:dict= None): # type: ignore
    tools= await client.get_tools()

    tool=next(
        t for t in tools if t.name==tool_name
    )

    result= await tool.ainvoke(
        tool_args or {}
    )

    return result
