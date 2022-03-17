# Imports
import streamlit as st
from pdt_info_extractor import get_sentences, get_entities


# Main App
"""
# PDT Information Extraction
"""
document_file = st.file_uploader("Upload PDT Document", type="pdf")
if document_file is not None:
    with open(document_file.name, "wb") as dfh:
        dfh.write(document_file.getbuffer())
        st.success("File Upload Successfull")

    with st.spinner("Extracting Info from the Document... Please Wait..."):
        sentences = get_sentences(document_file.name)
        entities = get_entities(sentences)
        sentences_dict = {i: sentence for i, sentence in enumerate(sentences)}
        entities['Context'] = entities.SentenceID.apply(lambda x: f"{sentences_dict.get(max(0, x-1))} | {sentences_dict.get(x)}")

    """
    ## Extracted Details
    """
    st.table(entities)
