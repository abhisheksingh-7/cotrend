import streamlit as st
import pydantic
import pickle

from langchain_core.prompts import ChatPromptTemplate
from langchain import output_parsers


class Output(pydantic.BaseModel):
    output: str


if "vecstore" not in st.session_state.keys():

    def load_vecstore():
        with open("/data/clapt-vecstore.pkl", "rb") as f:
            vec = pickle.load(f)
        return vec

    st.session_state["vecstore"] = load_vecstore()

vecstore = st.session_state["vecstore"]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user's question as a trained medical doctor."),
        ("user", "{input}"),
    ]
)
output_parser: output_parsers.PydanticOutputParser[Output] = (
    output_parsers.PydanticOutputParser(pydantic_object=Output)
)


# Set the page configuration for Streamlit
st.set_page_config(page_title="CLAPT RAG", layout="wide")
st.markdown(
    "<style>body { color: #fff; background-color: #111; }</style>",
    unsafe_allow_html=True,
)


def get_answer(question):
    """Get answer for the question using LangChain."""
    formatted_prompt = prompt.format_prompt(input=question).to_string()
    return formatted_prompt


def main():
    """Main function to run the Streamlit app."""
    # Sidebar settings
    st.sidebar.title("Q&A Settings")
    st.sidebar.info(
        "This app uses your own language model to answer questions based on the provided context."
    )

    # Main content area
    st.title("Llama-3-Clapt RAG Question Answering")
    st.write("Type your question.")

    # Context area
    # default_context = "Contrastively LeArned Perceptron Encoder from Decoders"
    # if "context" not in st.session_state:
    #     st.session_state["context"] = default_context

    # context = st.text_area("Context", value=st.session_state["context"], height=150)
    # st.session_state["context"] = context

    question = st.text_input("Question", "")

    # Button to get the answer
    if st.button("Get Answer"):
        with st.spinner("Searching for answers..."):
            answer = get_answer(question)
            st.write("**Answer:**", answer)


if __name__ == "__main__":
    main()
