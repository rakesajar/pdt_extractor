# Imports
import streamlit as st
from pdt_info_extractor_new import get_sentences, get_entities, get_information

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
        information = get_information(entities)

    """
    ### Extracted Information
    """
    st.write(information)
    """
    ### Extracted Entities
    """
    st.table(entities.iloc[:, :-1])
