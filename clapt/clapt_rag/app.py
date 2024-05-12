import streamlit as st
import pydantic

from langchain_core.prompts import ChatPromptTemplate
from langchain import output_parsers


class Output(pydantic.BaseModel):
    output: str


if "model" not in st.session_state.keys():
    st.session_state["model"] = loadPretrainedModel(embed_size, loss_type)
model = st.session_state["model"]

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
st.set_page_config(page_title="CLAPT RAG", layout="wide")
st.markdown(
    "<style>body { color: #fff; background-color: #111; }</style>",
    unsafe_allow_html=True,
)


def get_answer(question, context):
    """Get answer for the question using LangChain."""
    return qa_chain.run(question=question, context=context)


def main():
    """Main function to run the Streamlit app."""
    # Sidebar settings
    st.sidebar.title("Q&A Settings")
    st.sidebar.info(
        "This app uses your own language model to answer questions based on the provided context."
    )

    # Main content area
    st.title("Llama-3-CLAPED RAG")
    st.subheader("Contrastively LeArned Perceptron Encoder from Decoders")
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


if __name__ == "__main__":
    main()
