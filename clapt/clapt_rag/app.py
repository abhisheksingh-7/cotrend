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
        with open("/data/clapt-vecstore-big-2.pkl", "rb") as f:
            vec = pickle.load(f)
        return vec

    st.session_state["vecstore"] = load_vecstore()

vecstore: VectorStore = st.session_state["vecstore"]

tokenizer: AutoTokenizer = vecstore.tokenizer

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]


# Set the page configuration for Streamlit
# st.set_page_config(page_title="CoTrEnD", layout="wide")
# st.markdown(
#     "<style>body { color: #fff; background-color: #111; }</style>",
#     unsafe_allow_html=True,
# )
st.set_page_config(page_title="CoTrEnD", layout="wide")
st.markdown(
    "<style>body { color: #fff; background-color: #111; } .fullScreenFrame > div {height: 100vh !important; display: flex; flex-direction: column; justify-content: space-between;}</style>",
    unsafe_allow_html=True,
)


def get_answer(question):
    """Get answer for the question using LangChain."""
    messages = [
        {
            "role": "system",
            "content": "You are a trained medical doctor and biomedical expert. Answer the user's question as such.",
        }
    ]
    match_ = vecstore.search(question, k=1)[0][0]
    document_text = f"Title: {match_.title}\nAbstract: {match_.abstract}\n\n"
    user_message = {
        "role": "user",
        "content": "Context: " + document_text + "\nQuestion: " + question,
    }
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


def fetch_research_abstract(entity) -> str:
    matches, _ = vecstore.search(entity, k=3)
    result = ""
    for match_ in matches:
        result += f"Title: {match_.title}\nAbstract: {match_.abstract}\n\n"
    return result


def main():
    """Main function to run the Streamlit app."""
    col1, col2 = st.columns((1, 2))

    with col1:
        st.image("clapt/clapt_rag/static/cotrend.webp", width=300)
        # st.image("path_to_your_logo.png", width=300)  # Adjust path and size as needed
        st.write("Encoders that embed the final hidden state from large decoder models")

    with col2:
        st.title("Contrastively Trained Encodings from Decoder")
        question = st.text_input("Ask a Biomedical Question")
        if st.button("Get Answer", key="1"):
            with st.spinner("Browsing PubMed..."):
                answer = get_answer(question)
                st.write("**Answer:**", answer)

        entity_name = st.text_input(
            "Enter a medical entity to fetch research abstracts against"
        )
        if st.button("Fetch Abstract", key="2"):
            with st.spinner("Fetching research abstract..."):
                abstract = fetch_research_abstract(entity_name)
                st.write("**Abstract for", entity_name + ":**", abstract)


if __name__ == "__main__":
    main()
