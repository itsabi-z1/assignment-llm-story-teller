import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

load_dotenv()


@st.cache_resource
def create_llm_agent(model="gpt-oss", reasoning=False, temperature=0.5) -> Runnable:
    llm = ChatOllama(model=model, reasoning=reasoning, temperature=temperature)

    system_prompt = """
    You are a creative story teller.
    Create a short story based on the user's input.
    
    Instructions:
    - The story should be engaging and imaginative.
    - Language of the story should be simple and easy to understand for all age groups.
    - The story should have a clear beginning, middle, and end.
    
    Story format:
    <title of the story>
    <story body>
    <moral of the story>
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{input}")
        ]
    )
    return prompt | llm | StrOutputParser()


if __name__ == "__main__":

    story_teller = create_llm_agent()

    ## set up Streamlit
    st.title("LLM based Story Generator")
    story_gener = st.radio("Choose a story genre:", ('Fantasy', 'Sci-Fi', 'Mystery', 'Romance'))
    characters = st.text_input("Enter characters count (e.g. 2-3), or give names: e.g. Alice, Bob:")
    number_of_paragraphs = st.slider("Select number of paragraphs:", 1, 5, 3)
    user_input = st.text_input("Describe your story idea:")

    enhanced_user_input = f"Genre: {story_gener}. Characters: {characters}. Number of paragraphs: {number_of_paragraphs}. Idea: {user_input}"

    if user_input:
        story = story_teller.invoke({"input": user_input})
        st.write(story)
