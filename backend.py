import asyncio
import os
import certifi
from dotenv import load_dotenv
from httpx import get
load_dotenv()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

#on windows os wemay get path issues with certifi, so we set the environment variable to the certifi path
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()


from langchain_core.messages import (AnyMessage, BaseMessage, HumanMessage, AIMessage, SystemMessage)
from langchain_groq import ChatGroq
#from tools.tavily_tool import tavily_search
#from tools.flight_tool import search_flights
from tools.mcp_client_test import tavily_mcp_search, aviation_mcp_call

def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set. Please add your Render PostgreSQL External Database URL.")

    if "sslmode=" not in database_url:
        seperator="&" if "?" in database_url else "?"
        database_url=f"{database_url}{seperator}sslmode=require"
    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please add your Groq API key to .env file.")

# LLM
llm=ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY, # type: ignore
    max_tokens=900,
    temperature=0.3# type: ignore
    )

#graph state
class TravelState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    user_query:str
    flight_result:str
    hotel_result:str
    itinerary:str
    llm_calls:int# how many llm calls we are doing

'''
def flight_agent(state:TravelState):
    query=state["user_query"]
    flight_data=aviation_mcp_call(query)

    return{
        "flight_result":flight_data,
        "messages":[AIMessage(content="Flight results fetched")],
        "llm_calls":state.get("llm_calls",0)+1
    }
'''
#-----------------------Flight Agent -------------------
FLIGHT_AGENT_PROMPT = """
    You are a travel flight expert.

    User Query: {query}
    Airport Information: {airport_data}
    Airline Information: {airline_data}

    Generate :
        1. Likely departure airport
        2. Likely arrival airport
        3. Airlines serving this route
        4. Typical flight duration
        5. Estimated airfare range
        6. Peak season pricing warning
        7. Booking advice

Return concise travel guidance.

"""

def flight_agent(state:TravelState):
    print("\n Inside Flight Agent")
    query = state["user_query"]

    try:
        airports = asyncio.run(
            aviation_mcp_call("list_airports")
        )

        airlines = asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:
        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_result": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }




#will do tavily search and give hotel information
def hotel_agent(state:TravelState):
    query=f"Best hotels for {state['user_query']}"
    #hotel_results=tavily_search(query)
    hotel_results=asyncio.run(tavily_mcp_search(query))
    return{
        "hotel_result":hotel_results,
        "messages":[AIMessage(content="Hotel results fetched")],
        "llm_calls":state.get("llm_calls",0)+1
    }

def itinerary_agent(state:TravelState):
    prompt=f"""

    Create a practical travel itinerary using the information below.

    User request:
    {state["user_query"]}

    Flight information:
    {state["flight_result"]}

    Hotel information:
    {state["hotel_result"]}

    Requirements:
    - Create a day-by-day itinerary.
    - Include major sightseeing activities.
    - Consider travel time between places.
    - Keep the plan realistic and budget-aware.
    - Mention useful hotel/flight details when available.
    - Do not invent unavailable flight prices or hotel information.
        
    """
    response=llm.invoke([
        SystemMessage(content="You are a expert travel assistant that creates detailed travel itineraries based on flight and hotel information."),
        HumanMessage(content=prompt)
    ])

    return{
        "itinerary":response.content,
        "messages":[response],
        "llm_calls":state.get("llm_calls",0)+1
    }

def final_agent(state:TravelState):
    prompt=f"""
    Create the final travel plan for the user.

        User request:
        {state["user_query"]}

        Flights:
        {state["flight_result"]}

        Hotels:
        {state["hotel_result"]}

        Itinerary:
        {state["itinerary"]}

        Format the response using:

        # Trip Summary
        # Flight Information
        # Hotel Suggestions
        # Weather Information
        # Day-by-Day Itinerary
        # Estimated Budget
        # Final Recommendations

        Rules:
        - Be practical and concise.
        - Do not invent flight prices.
        - Clearly state when live pricing is unavailable.
        - Include weather/travel advice only when supported by available information.
        - Prefer useful information over unnecessary explanation.
    """
    response=llm.invoke([
        SystemMessage(content="You are a expert travel assistant that creates final responses based on flight, hotel and itinerary information."),
        HumanMessage(content=prompt)
    ])

    return{
        "messages":[response],
        "llm_calls":state.get("llm_calls",0)+1
    }


#--------------------------State Graph-----------------------------
# Create a state graph for the travel planning process
graph = StateGraph(TravelState)

graph.add_node("flight_agent",flight_agent)
graph.add_node("hotel_agent",hotel_agent)
graph.add_node("itinerary_agent",itinerary_agent)
graph.add_node("final_agent",final_agent)

graph.add_edge(START,"flight_agent")
graph.add_edge("flight_agent","hotel_agent")
graph.add_edge("hotel_agent","itinerary_agent")
graph.add_edge("itinerary_agent","final_agent")
graph.add_edge("final_agent",END)

#from IPython.display import Image,display
#display(Image(backend.get_graph().draw_mermaid_png()))


#postgresql checkpoint
DATABASE_URL=get_database_url()

_conn=psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row # type: ignore
)
#_conn this connection is passed inside the checkpointer
checkpointer=PostgresSaver(_conn) # type: ignore
checkpointer.setup()

#pass this checkpointer inside graph
travel_graph=graph.compile(checkpointer=checkpointer)

#all these states are saved inside the memory

#function for fastapi
def run_travel_agent(user_input:str, thread_id:str | None=None):
    if not thread_id:
        thread_id=f"user_{uuid.uuid4().hex}"

    config={
        "configurable":{
            "thread_id":thread_id
        }
    }

    result=travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_result":"",
            "hotel_result":"",
            "itinerary":"",
            "llm_calls": 0,
        },
        config=config # type: ignore
    )

    final_answer=result["messages"][-1].content

    return{
        "thread_id":thread_id,
        "answer":final_answer,
        "flight_results":result.get("flight_result",""),
        "hotel_results":result.get("hotel_result",""),
        "itinerary":result.get("itinerary",""),
        "llm_calls":result.get("llm_calls",0),
    }