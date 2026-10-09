'''

            PHASE 1 Testing


from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent
#result=tavily_search("Best hotels in India")
#print(result)

#es=search_flights("Plan a 7 days Japan trip from India")
#rint(res)
user_input=input("Enter travel request:")
res=run_travel_agent(
    user_input=user_input,
    thread_id="test_user"
)

print("\n FINAL RESPONSE: \n")
print(res['answer'])


'''
import asyncio
from tools.mcp_client_test import get_all_tools, tavily_mcp_search

if __name__=="__main__":
    #this is asynchronous funtion so to run it we need asyncio

    #asyncio.run(get_all_tools())
    #query="latest news about AI"
    #asyncio.run(tavily_mcp_search(query)) # type: ignore
    asyncio.run(get_all_tools())