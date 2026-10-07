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
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

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
    api_key=GROQ_API_KEY # type: ignore
    )

#graph state
class TravelState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    user_query:str
    flight_result:str
    hotel_result:str
    itinerary:str
    llm_calls:int# how many llm calls we are doing


def flight_agent(state:TravelState):
    query=state["user_query"]
    flight_data=search_flights(query)

    return{
        "flight_result":flight_data,
        "messages":[AIMessage(content="Flight results fetched")],
        "llm_calls":state.get("llm_calls",0)+1
    }

#will do tavily search and give hotel information
def hotel_agent(state:TravelState):
    query=f"Best hotels for {state['user_query']}"
    hotel_results=tavily_search(query)

    return{
        "hotel_result":hotel_results,
        "messages":[AIMessage(content="Hotel results fetched")],
        "llm_calls":state.get("llm_calls",0)+1
    }

def itinerary_agent(state:TravelState):
    prompt=f"""

    You are a travel assistant. You have been given the following information about flights and hotels for a user's travel query. 
    Create a detailed travel itinerary based on the provided information.:

    User Query: {state['user_query']}
    Flight Results: {state['flight_result']}
    Hotel Results: {state['hotel_result']}

    Make the itinerary practical,informative, engaging, budget aware and easy to follow. Include details such as flight timings, hotel amenities, and any other relevant information that would enhance the user's travel experience.
    
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
    You are a travel assistant. You have been given the following information about flights, hotels, and a detailed itinerary for a user's travel query. 
    Create a final response that is engaging, informative, and easy to understand. Make sure to highlight the key points from the itinerary and provide any additional tips or recommendations that would enhance the user's travel experience.

    User Query: {state['user_query']}
    Flight Results: {state['flight_result']}
    Hotel Results: {state['hotel_result']}
    Itinerary: {state['itinerary']}

    Format the final answer beautifully using these sections:

        1. Trip Summary
        2. Flight Information
        3. Hotel Suggestions
        4. Weather Information
        5. Day-by-Day Itinerary
        6. Estimated Budget
        7. Final Recommendations

    Important:
    - Be clear and practical.
    - Mention that live flight API may not provide ticket prices if pricing is unavailable.
    - Include weather-based travel advice.
    - Keep the response useful for real travel planning.

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
        thread_id=f"user_{uuid.uuid4.hex}"

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
        "flight_results":result.get("flight_results",""),
        "hotel_results":result.get("hotel_results",""),
        "intinerary":result.get("itinerary",""),
        "llm_calls":result.get("llm_calls",0),
    }