# Imports
import re
import fitz
import tner
import pandas as pd


# Globals
tner_tagger = tner.TransformersNER('asahi417/tner-xlm-roberta-base-ontonotes5')


# Functions
def get_sentences(document_fname):
    document = fitz.open(document_fname)
    sentences = []
    for page in document:
        page_text = page.get_text()
        page_sentences = page_text.split("\n")
        page_sentences = [sentence.strip() for sentence in page_sentences]
        sentences.extend(page_sentences)
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip())>0]
    return sentences


def _get_gender(list_of_strings):
    for id, sentence in enumerate(list_of_strings):
        list_of_words = sentence.lower().split()
        if 'female' in list_of_words:
            return pd.DataFrame({"SentenceID": [id], "EntityType": ['gender'], "StartPosition": [sentence.lower().find("female")], "EndPosition": [sentence.lower().find("female")+6], "EntityText": ['Female'], "ConfidenceScore": [1.0]})
        if 'male' in list_of_words:
            return pd.DataFrame({"SentenceID": [id], "EntityType": ['gender'], "StartPosition": [sentence.lower().find("male")], "EndPosition": [sentence.lower().find("male")+4], "EntityText": ['Male'], "ConfidenceScore": [1.0]})
    return None


def _get_test_result(list_of_strings):
    for id, sentence in enumerate(list_of_strings):
        filtered_sentence = ''.join(filter(str.isalpha, sentence))
        if 'negative' in filtered_sentence.lower():
            return pd.DataFrame({"SentenceID": [id], "EntityType": ['test_result'], "StartPosition": [sentence.lower().find("negative")], "EndPosition": [sentence.lower().find("negative")+8], "EntityText": ['Negative'], "ConfidenceScore": [1.0]})
        if 'positive' in filtered_sentence.lower():
            return pd.DataFrame({"SentenceID": [id], "EntityType": ['test_result'], "StartPosition": [sentence.lower().find("positive")], "EndPosition": [sentence.lower().find("positive")+8], "EntityText": ['Positive'], "ConfidenceScore": [1.0]})
    return None


def _get_test_type(list_of_strings):
    for id, sentence in enumerate(list_of_strings):
        filtered_sentence = ''.join(filter(str.isalpha, sentence))
        filtered_sentence = filtered_sentence.lower()
        if 'rtpcr' in filtered_sentence:
            return pd.DataFrame({"SentenceID": [id], "EntityType": ['test_type'], "StartPosition": [sentence.lower().find("rt")], "EndPosition": [sentence.lower().find("rt")+6], "EntityText": ['RT-PCR'], "ConfidenceScore": [1.0]})
        if 'pcr' in filtered_sentence:
            return pd.DataFrame({"SentenceID": [id], "EntityType": ['test_type'], "StartPosition": [sentence.lower().find("pcr")], "EndPosition": [sentence.lower().find("pcr")+6], "EntityText": ['PCR'], "ConfidenceScore": [1.0]})
        if 'rapid' in filtered_sentence:
            if 'antigen' in filtered_sentence:
                return pd.DataFrame({"SentenceID": [id], "EntityType": ['test_type'], "StartPosition": [sentence.lower().find("rapid")], "EndPosition": [sentence.lower().find("rapid")+6], "EntityText": ['Rapid Antigen'], "ConfidenceScore": [1.0]})
    return None


def _get_passport_number(list_of_strings):
    passport_dfs = []
    for id, sentence in enumerate(list_of_strings):
        passport_candidates = re.finditer(r"\b[A-Z1-9]\w[0-9]{4}\d?\d?\d?\w?\b", sentence, re.MULTILINE)
        for passport_candidate in passport_candidates:
            passport_candidate = passport_candidate.group()
            if 7<= len(passport_candidate) <= 10:
                conf_score = 0.9 if len(passport_candidate) == 8 else 0.7
                pdf = pd.DataFrame({"SentenceID": [id], "EntityType": ['passport_number'], "StartPosition": [sentence.lower().find(passport_candidate)], "EndPosition": [sentence.lower().find(passport_candidate)+len(passport_candidate)], "EntityText": [passport_candidate], "ConfidenceScore": [conf_score]})
                passport_dfs.append(pdf)
    if len(passport_dfs)>0:
        passport_df = pd.concat(passport_dfs)
        return passport_df
    return None


def get_entities(list_of_strings):
    sentence_id, category, start_pos, end_pos, text, confidence_score = [], [], [], [], [], []
    for id, sentence in enumerate(list_of_strings):
        result = tner_tagger.predict([sentence])[0]['entity']
        if len(result)>0:
            for item in result:
                sentence_id.append(id)
                category.append(item['type'])
                start_pos.append(item['position'][0])
                end_pos.append(item['position'][1])
                text.append(item['mention'])
                confidence_score.append(item['probability'])
    entities = pd.DataFrame({"SentenceID": sentence_id, "EntityType": category, "StartPosition": start_pos, "EndPosition": end_pos, "EntityText": text, "ConfidenceScore": confidence_score})
    entities = entities[entities.EntityType.isin(["person", "date", "time"])]
    # entities = entities.groupby('SentenceID').agg(",".join).reset_index()
    # Get Gender
    gender_df = _get_gender(list_of_strings)
    if gender_df is not None:
        entities = pd.concat([entities, gender_df])
    # Get Test Result
    test_result_df = _get_test_result(list_of_strings)
    if test_result_df is not None:
        entities = pd.concat([entities, test_result_df])
    # Get Test Type
    test_type_df = _get_test_type(list_of_strings)
    if test_type_df is not None:
        entities = pd.concat([entities, test_type_df])
    # Get Passport Number
    passport_df = _get_passport_number(list_of_strings)
    if passport_df is not None:
        entities = pd.concat([entities, passport_df])
    entities = entities.sort_values("SentenceID")
    entities['EntityType'] = entities.EntityType.str.upper()
    return entities