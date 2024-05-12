import streamlit as st
import pydantic

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.chat_models import ChatOllama
from langchain import output_parsers


class Output(pydantic.BaseModel):
    output: str


llm = Ollama(model="llama3")
chat_model = ChatOllama()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user's question as a trained medical doctor."),
        ("user", "{input}"),
    ]
)
output_parser: output_parsers.PydanticOutputParser[Output] = (
    output_parsers.PydanticOutputParser(pydantic_object=Output)
)

qa_chain = prompt | chat_model | output_parser


# Set the page configuration for Streamlit
st.set_page_config(page_title="Llama-3 CLAPT RAG", layout="wide")
st.markdown(
    "<style>body { color: #fff; background-color: #111; }</style>",
    unsafe_allow_html=True,
)


def get_answer(question, context):
    """Get answer for the question using LangChain."""
    return qa_chain.invoke(input=question)


def fetch_research_abstract(entity):
    abstract = "Simulated abstract content for " + entity
    return abstract


def main():
    """Main function to run the Streamlit app."""
    # Sidebar settings
    st.sidebar.title("Q&A Settings")
    st.sidebar.info(
        "This app uses your own language model to answer questions based on the provided context."
    )

    # Main content area
    st.title("Llama-3 CLAPT RAG")
    st.write("Type your question.")

    # Context area
    default_context = "Streamlit is an open-source app framework developed in Python. It is used to turn data scripts into shareable web apps in minutes. All you need to do is write your script and run it with Streamlit."
    if "context" not in st.session_state:
        st.session_state["context"] = default_context

    context = st.text_area("Context", value=st.session_state["context"], height=150)
    st.session_state["context"] = context

    question = st.text_input("Question", "What is Streamlit?")
    # Button to get the answer
    if st.button("Get Answer"):
        with st.spinner("Searching for answers..."):
            answer = get_answer(question, context)
            st.write("**Answer:**", answer)

    entity_name = st.text_input(
        "Medical Entity", "Enter a medical entity like fever or cough"
    )
    # Button and expander for research abstract
    if st.button("Fetch Abstract"):
        with st.spinner("Fetching research abstract..."):
            abstract = fetch_research_abstract(entity_name)
            expander = st.expander("Research Abstract for " + entity_name)
            expander.write(abstract)


if __name__ == "__main__":
    main()
