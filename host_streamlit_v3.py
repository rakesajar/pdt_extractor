# Imports
import streamlit as st
from pdt_info_extractor_v3 import get_sentences, get_entities, get_information

# Main App
"""
# PDT Information Extraction
"""
document_file = st.file_uploader("Upload PDT Document", type="pdf")
if document_file is not None:
    with open(document_file.name, "wb") as dfh:
        dfh.write(document_file.getbuffer())
        st.success("File Upload Successful")

    with st.spinner("Extracting Info from the Document... Please Wait..."):
        # sentences = get_sentences(document_file.name)
        sentences = get_sentences(open(document_file.name, 'rb').read())
        entities = get_entities(sentences)
        information = get_information(entities)
        result = {"_".join(k.lower().split()): v for k, v in information.items() if not type(v) == list}

    """
    ### Extracted Information
    """
    st.write(result)
    """
    ### Extracted Entities
    """
    if st.checkbox("Show Entity Context"):
        st.table(entities.iloc[:, :])
    else:
        st.table(entities.iloc[:, :-1])
