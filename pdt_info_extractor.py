import fitz
import spacy
import unicodedata
import pandas as pd
import streamlit as st


# Globals
nlp = spacy.load("en_core_web_sm")
trf = spacy.load("en_core_web_trf")


# Classes
class PDFdocument:
    def __init__(self, fname):
        self.fname = fname
        self.doc = fitz.open(fname)
        self.page_count = self.doc.page_count
        self.meta_data = self.doc.metadata


def extract_info(sentences):
    name, gender, collection, specimen, test_type, result = None, None, None, None, None, None
    for sentence in sentences:
        if sentence.lower().startswith("name:"):
            if not name:
                name = sentence.lower().split("name:")[-1].upper().strip()
        if not gender:
            if 'female' in sentence.lower().split():
                gender = 'FEMALE'
            elif 'male' in sentence.lower().split():
                gender = 'MALE'
        if not result:
            if 'negative' in sentence.lower().split():
                result = 'NEGATIVE'
            if '(negative)' in sentence.lower().split():
                result = 'NEGATIVE'
            if 'positive' in sentence.lower().split():
                result = 'POSITIVE'
            if '(positive)' in sentence.lower().split():
                result = 'POSITIVE'
        if not specimen:
            if 'swab' in sentence.lower():
                specimen = sentence.split(":")[-1].strip()
                specimen = specimen.split("swab")[0].strip() + ' swab'
        if not test_type:
            if 'rt' in sentence.lower():
                if 'pcr' in sentence.lower():
                    test_type = 'RT-PCR'
            elif 'pcr' in sentence.lower():
                test_type = 'PCR'
            elif 'rapid' in sentence.lower():
                if 'antigen' in sentence.lower():
                    test_type = 'RAPID ANTIGEN'
        if not collection:
            if 'collect' in sentence.lower():
                trf_doc = trf(sentence)
                candidate_date = [ent.text for ent in trf_doc.ents if ent.label_ == 'DATE']
                if candidate_date:
                    collection = candidate_date[0]

    result = {'Name': name,
              'Gender': gender,
              'Specimen': specimen,
              'Collection Date/Time': collection,
              'Test Type': test_type,
              'Result': result}
    return result

def get_entities(sentences):
    sentence_ids, entity_text, entity_label = [], [], []
    for sentence_id, sentence in enumerate(sentences):
        spacy_doc = trf(sentence)
        for ent in spacy_doc.ents:
            entity_text.append(ent.text)
            entity_label.append(ent.label_)
            sentence_ids.append(sentence_id)
        entities = pd.DataFrame({'Text': entity_text, 'Entity': entity_label, 'Sentence': sentence_ids})
    return entities

def get_sentences(fname):
    document = PDFdocument(fname)
    sentences = []
    for i, page in enumerate(document.doc):
        blocks = page.get_text("blocks")
        blocks = [block[4] for block in blocks]
        sentences.extend(blocks)
        sentences = [sent for sentence in sentences for sent in sentence.split("\n")]
        sentences = [unicodedata.normalize("NFKD", unicode_str) for unicode_str in sentences]
        sentences = [sentence for sentence in sentences if not sentence.startswith("<image:")]
        sentences = [sentence.strip() for sentence in sentences if len(sentence.strip())>2]
        return sentences

def get_info_subset1(entities, sentences):
    name, dob, collect = None, None, None
    entities.to_dict(orient="records")
    for item in entities.to_dict(orient="records"):
        if item['Entity'] == 'PERSON':
            for i in range(max(0, item['Sentence']-1), min(item['Sentence']+2, len(sentences))):
                if 'name' in sentences[i].lower():
                    if not name:
                        name = item['Text']
        if item['Entity'] in ["TIME", "DATE"]:
            for i in range(max(0, item['Sentence']-1), min(item['Sentence']+2, len(sentences))):
                if 'birth' in sentences[i].lower():
                    if not dob:
                        dob = item['Text']
        if item['Entity'] in ["TIME", "DATE"]:
            for i in range(max(0, item['Sentence']-1), min(item['Sentence']+2, len(sentences))):
                if 'collect' in sentences[i].lower():
                    if not collect:
                        collect = item['Text']
    return {'Name': name, 'Date of Birth': dob, 'Collection Date/Time': collect}


def get_info_subset2(sentences):
    name, gender, collection, specimen, test_type, result = None, None, None, None, None, None
    for sentence in sentences:
        if sentence.lower().startswith("name:"):
            if not name:
                name = sentence.lower().split("name:")[-1].upper().strip()
        if not gender:
            if 'female' in sentence.lower().split():
                gender = 'FEMALE'
            elif 'male' in sentence.lower().split():
                gender = 'MALE'
        if not result:
            if 'negative' in sentence.lower().split():
                result = 'NEGATIVE'
            if '(negative)' in sentence.lower().split():
                result = 'NEGATIVE'
            if 'positive' in sentence.lower().split():
                result = 'POSITIVE'
            if '(positive)' in sentence.lower().split():
                result = 'POSITIVE'
        if not specimen:
            if 'swab' in sentence.lower():
                specimen = sentence.split(":")[-1].strip()
                specimen = specimen.lower().split("swab")[0].strip() + ' swab'
        if not test_type:
            if 'rt' in sentence.lower():
                if 'pcr' in sentence.lower():
                    test_type = 'RT-PCR'
            elif 'pcr' in sentence.lower():
                test_type = 'PCR'
            elif 'rapid' in sentence.lower():
                if 'antigen' in sentence.lower():
                    test_type = 'RAPID ANTIGEN'
        if not collection:
            if 'collect' in sentence.lower():
                trf_doc = trf(sentence)
                candidate_date = [ent.text for ent in trf_doc.ents if ent.label_ == 'DATE']
                if candidate_date:
                    collection = candidate_date[0]

    result = {'Name': name,
              'Gender': gender,
              'Specimen': specimen,
              'Collection Date/Time': collection,
              'Test Type': test_type,
              'Result': result}
    return result

def get_info(fname):
    sentences = get_sentences(fname)
    entities = get_entities(sentences)
    result1 = get_info_subset1(entities, sentences)
    result2 = get_info_subset2(sentences)
    result = result2 | result1
    return sentences, entities, result

"""
# PDT Info Extraction
"""
document_file = st.file_uploader("Upload PDT Document", type="pdf")
if document_file is not None:
    with open(document_file.name, "wb") as dfh:
        dfh.write(document_file.getbuffer())
        st.success("File Upload Successfull")

    with st.spinner("Extracting information from the document... Please wait."):
        sentences, entities, result = get_info(document_file.name)
    st.write(result)

    if st.checkbox("Show All Sentences"):
        st.write(sentences)

    if st.checkbox("Show All Entities"):
        st.write(entities.to_dict(orient="records"))

