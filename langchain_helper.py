from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.6
)


def generate_restaurant_name_and_items(cuisine):  # ✅ Use parameter!

    # Chain 1: Cuisine → Restaurant Name
    name_prompt = PromptTemplate.from_template(
        "Suggest a fancy restaurant name for {cuisine} food."
    )
    chain1 = name_prompt | llm | StrOutputParser()

    restaurant_name = chain1.invoke({"cuisine": cuisine})  # ✅ Use cuisine param!

    # Chain 2: Restaurant Name → Menu Items
    items_prompt = PromptTemplate.from_template(
        """Suggest 8 signature menu items for the '{restaurant_name}' restaurant. 
        Return as comma-separated list only."""
    )
    chain2 = items_prompt | llm | StrOutputParser()

    menu_items = chain2.invoke({"restaurant_name": restaurant_name})  # ✅ Pass result!

    return {
        "cuisine": cuisine,
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }


if __name__ == '__main__':
    result = generate_restaurant_name_and_items('Arabic')
    print(f"Cuisine: {result['cuisine']}")
    print(f"Restaurant: {result['restaurant_name']}")
    print(f"Menu: {result['menu_items']}")
