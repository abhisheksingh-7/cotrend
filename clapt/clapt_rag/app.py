from typing import Literal
import streamlit as st
import pydantic
import pickle

from langchain_core.prompts import ChatPromptTemplate
from clapt.clapt_rag import generate_vecstore
from langchain import output_parsers
from transformers import AutoTokenizer, AutoModelForCausalLM


from clapt.clapt_rag.generate_vecstore import VectorStore


Role = Literal["system", "user", "assistant"]


if "vecstore" not in st.session_state.keys():

    def load_vecstore():
        with open("/data/clapt-vecstore.pkl", "rb") as f:
            vec = pickle.load(f)
        return vec

    st.session_state["vecstore"] = load_vecstore()

<<<<<<< HEAD
vecstore: generate_vecstore.VectorStore = st.session_state["vecstore"]
=======
vecstore: VectorStore = st.session_state["vecstore"]
>>>>>>> master

tokenizer: AutoTokenizer = vecstore.tokenizer

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]


# Set the page configuration for Streamlit
st.set_page_config(page_title="Llama-3 CLAPT RAG", layout="wide")
st.markdown(
    "<style>body { color: #fff; background-color: #111; }</style>",
    unsafe_allow_html=True,
)


def get_answer(question):
    """Get answer for the question using LangChain."""
    # formatted_prompt = prompt.format_prompt(input=question).to_string()
    messages = [
        {
            "role": "system",
            "content": "You are a trained medical doctor and biomedical expert. Answer the user's question as such.",
        }
    ]
    user_message = {"role": "user", "content": question}
    messages.append(user_message)
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(vecstore.device)
    model = vecstore.embedding_model.decoder_with_lm
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1] :]
    output = tokenizer.decode(response, skip_special_tokens=True)
    print(output)

    return output


def fetch_research_abstract(entity):
    return vecstore.search(entity, k=1)[0][0].text


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

    question = st.text_input("Question", "")

    # Button to get the answer
    if st.button("Get Answer"):
        with st.spinner("Searching for answers..."):
            answer = get_answer(question)
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
