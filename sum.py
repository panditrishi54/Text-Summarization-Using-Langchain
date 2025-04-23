from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 


summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn",
    framework="pt"
)

llm = HuggingFacePipeline(pipeline=summarizer)

template = """
Summarize the following text in 2-3 sentences, capturing the main points:

{text}

Summary:
"""
prompt = PromptTemplate(input_variables=["text"], template=template)

summarization_chain = LLMChain(llm=llm, prompt=prompt)


def summarize_text(input_text):
    try:
        if not input_text.strip():
            return "Error: Input text is empty."
        result = summarization_chain.run(text=input_text)
        return result.strip()
    except Exception as e:
        return f"Error during summarization: {str(e)}"

# Flask API endpoint for summarization
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get('text', '')
    summary = summarize_text(text)
    return jsonify({'summary': summary})

# Example usage for command-line testing
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)