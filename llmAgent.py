from langchain_classic.chains.llm import LLMChain  # Keep this if you want
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Your app import (assuming it has cuisine variable)
from app import cuisine
load_dotenv()

# Groq model
print('Invoking groq model')
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.6
)

# Test single call
name = llm.invoke('I want to open a restaurant for italian food. Suggest a fancy name for it.')
print("Restaurant name:", name.content)

# Fix typo: cusine → cuisine
prompt_template_name = PromptTemplate(
    input_variables=['cuisine'],  # Fixed typo
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for it."
)
print(prompt_template_name.format(cuisine='Indian'))  # Fixed variable name

### CHAIN 1: Cuisine → Restaurant Name
print("\nChain 1 - Restaurant Name:")
chain1 = prompt_template_name | llm
restaurant_name = chain1.invoke({"cuisine": "Indian"}).content  # Use dict input
print(restaurant_name)

### CHAIN 2: Restaurant Name → Menu Items
prompt_template_items = PromptTemplate(
    input_variables=['restaurant_name'],
    template="Suggest some menu items for '{restaurant_name}' restaurant. Return as comma separated."
)

chain2 = prompt_template_items | llm
menu_items = chain2.invoke({"restaurant_name": restaurant_name}).content  # Pass restaurant_name
print("\nChain 2 - Menu Items:")
print(menu_items)

### SEQUENTIAL CHAIN (LCEL): Cuisine → Name → Menu
print("\n=== FULL SEQUENTIAL CHAIN ===")
# Proper LCEL sequential chain
sequential_chain = (
    prompt_template_name 
    | llm 
    | PromptTemplate.from_template("Suggest menu items for '{restaurant_name}' restaurant as comma-separated list.")
    | llm
)

result = sequential_chain.invoke({"cuisine": "Arabic"})
print("Sequential result:", result.content)
