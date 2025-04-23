import streamlit as st
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Function to generate summary
def generate_response(txt, api_key, chain_type="map_reduce", chunk_size=1000, temperature=0):
    """
    Generate a summary for the input text using LangChain and OpenAI.
    
    Args:
        txt (str): Input text to summarize.
        api_key (str): OpenAI API key.
        chain_type (str): Type of summarization chain ('map_reduce', 'stuff', 'refine').
        chunk_size (int): Size of text chunks for splitting.
        temperature (float): LLM temperature for controlling creativity.
    
    Returns:
        str: Summary or error message.
    """
    try:
        # Instantiate the LLM model
        llm = OpenAI(temperature=temperature, openai_api_key=api_key)
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        texts = text_splitter.split_text(txt)
        # Create documents
        docs = [Document(page_content=t) for t in texts]
        # Text summarization
        chain = load_summarize_chain(llm, chain_type=chain_type)
        return chain.run(docs)
    except Exception as e:
        return f"Error: {str(e)}"

# Page configuration
st.set_page_config(page_title="🦜🔗 Text Summarization App", layout="wide")
st.title("🦜🔗 Text Summarization App")

# Text input
st.subheader("Input Text")
txt_input = st.text_area("Enter the text you want to summarize", "", height=200)
st.caption(f"Character count: {len(txt_input)}")

# Advanced options
with st.expander("Advanced Settings"):
    chain_type = st.selectbox("Summarization Chain Type", ["map_reduce", "stuff", "refine"], 
                              help="Choose how the text is summarized. 'map_reduce' is good for long texts.")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100,
                           help="Size of text chunks for processing. Smaller chunks are more precise but slower.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1,
                            help="Controls creativity. 0.0 is deterministic, 1.0 is more creative.")

# Form for summarization
with st.form("summarize_form", clear_on_submit=False):
    openai_api_key = st.text_input("OpenAI API Key", type="password", 
                                   help="sk-proj--ZBg5rzxcwvmmrAtGygiYdlpLs0ToAeSbKb1h6JyZnZjRfMPXp0LSPZFSUqm7FVfAYQDrgqMTLnyYLzYJHMNuX91IJyfQ6IMJ7B28ijpN8A")
    submitted = st.form_submit_button("Summarize")
    
    if submitted:
        if not txt_input.strip():
            st.error("Please enter text to summarize.")
        elif not openai_api_key:
            st.error("Please provide an OpenAI API key.")
        else:
            with st.spinner("Generating summary..."):
                response = generate_response(txt_input, openai_api_key, chain_type, chunk_size, temperature)
                if response.startswith("Error"):
                    st.error(response)
                else:
                    st.success("Summary generated successfully!")
                    st.markdown(f"**Summary**: {response}")

# Instructions for getting an OpenAI API key
st.subheader("How to Get an OpenAI API Key")
st.markdown("""
To use this app, you need an OpenAI API key. Follow these steps to obtain one:
1. Visit [OpenAI API Keys](https://platform.openai.com/account/api-keys).
2. Sign in or create an OpenAI account.
3. Click **+ Create new secret key**.
4. (Optional) Name your key for reference, then click **Create secret key**.
5. Copy the key (starts with `sk-`) and paste it above.

**Note**: Keep your API key secure and do not share it publicly. For production apps, store it in environment variables or Streamlit secrets.
""")

# Security warning
st.warning("For security, avoid hardcoding your API key in the code. Use Streamlit secrets or environment variables in production.")