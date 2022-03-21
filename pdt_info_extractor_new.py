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


def _get_context(list_of_strings, entities_df):
    sentences_dict = {i: sentence for i, sentence in enumerate(list_of_strings)}
    entities_df['Context'] = entities_df.SentenceID.apply(
        lambda x: f"{sentences_dict.get(max(0, x - 1))} | {sentences_dict.get(x)}")
    return entities_df


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
    entities = entities.sort_values(["SentenceID", "StartPosition"])
    entities['EntityType'] = entities.EntityType.str.upper()
    # Get Context
    entities = _get_context(list_of_strings, entities)
    entities.to_csv("entities.csv", index=False)
    return entities


def _get_name_info(entities):
    all_names = entities[entities.EntityType == 'PERSON'].EntityText.tolist()
    all_names_context = entities[entities.EntityType == 'PERSON'].Context.tolist()
    name, first_name, middle_name, last_name = "", "", "", ""
    for name, context in zip(all_names, all_names_context):
        cleaned_context = re.sub(r'[^A-Za-z0-9 ]+', '', context).lower()
        if 'first name' in cleaned_context.lower():
            first_name = name
        elif 'middle name' in cleaned_context.lower():
            middle_name = name
        elif 'last name' in cleaned_context.lower():
            last_name = name
        elif 'name' in cleaned_context.lower():
            name = name
    if any([first_name, middle_name, last_name]):
        name = f"{first_name} {middle_name} {last_name}"
        name = name.replace("  ", " ")
        name = name.strip()

    return name, all_names


def _get_dob_info(entities):
    all_dates = entities[entities.EntityType == 'DATE'].EntityText.tolist()
    dob = [date for date in all_dates if '2022' not in date][0]
    return dob, all_dates


def _get_passport_info(entities):
    all_passport_numbers = entities[entities.EntityType == 'PASSPORT_NUMBER'].EntityText.tolist()
    all_passport_context = entities[entities.EntityType == 'PASSPORT_NUMBER'].Context.tolist()
    passport_number = ""
    for pno, pc in zip(all_passport_numbers, all_passport_context):
        cleaned_context = re.sub(r'[^A-Za-z0-9 ]+', '', pc).lower()
        if 'passport' in cleaned_context:
            passport_number = pno
    return passport_number, all_passport_numbers


def _get_sample_date_time(entities):
    all_dates_sentence_ids = entities[entities.EntityType == 'DATE'].SentenceID.tolist()
    all_dates = entities[entities.EntityType == 'DATE'].EntityText.tolist()
    all_dates_context = entities[entities.EntityType == 'DATE'].Context.tolist()
    sample_date = ""
    sample_date_sentence_id = ""
    for sid, date, context in zip(all_dates_sentence_ids, all_dates, all_dates_context):
        cleaned_context = re.sub(r'[^A-Za-z0-9 ]+', '', context).lower()
        if 'sampling' in cleaned_context:
            sample_date = date
            sample_date_sentence_id = sid
        elif 'taken on' in cleaned_context:
            sample_date = date
            sample_date_sentence_id = sid
        elif 'collected' in cleaned_context:
            sample_date = date
            sample_date_sentence_id = sid
    all_potential_sample_dates = [date for date in all_dates if '2022' in date]
    sample_time = ""
    if len(sample_date)>0:
        all_times = entities[entities.EntityType == 'TIME'].EntityText.tolist()
        all_potential_sample_times = entities[entities.EntityType == 'TIME']
        all_potential_sample_times = all_potential_sample_times[all_potential_sample_times.SentenceID == sample_date_sentence_id].EntityText.tolist()
        sample_time = all_potential_sample_times[0] if len(all_potential_sample_times)>0 else ""
    return sample_date, all_potential_sample_dates, sample_time, all_times


def get_information(entities):
    # Get Name
    name, all_names = _get_name_info(entities)
    # Get Date of Birth
    dob, all_dates = _get_dob_info(entities)
    all_potential_dobs = [date for date in all_dates if '2022' not in date]
    # Get Gender
    gender_list = entities[entities.EntityType == 'GENDER'].EntityText.tolist()
    gender = gender_list[0] if len(gender_list)>0 else "NA"
    # Get Passport
    passport_number, all_passport_numbers = _get_passport_info(entities)
    # Test Type
    test_type_list = entities[entities.EntityType == 'TEST_TYPE'].EntityText.tolist()
    test_type = test_type_list[0] if len(test_type_list)>0 else "NA"
    # Test Result
    test_result_list = entities[entities.EntityType == 'TEST_RESULT'].EntityText.tolist()
    test_result = test_result_list[0] if len(test_result_list) > 0 else "NA"
    # Test Sample Collection Date and Time
    sample_date, all_potential_sample_dates, sample_time, all_potential_sample_times = _get_sample_date_time(entities)
    # Return Dict
    result = {"Name": name,
              "Date of Birth": dob,
              "Gender": gender,
              "Passport Number": passport_number,
              "Test Type": test_type,
              "Test Result": test_result,
              "Test Sample Date": sample_date,
              "Test Sample Time": sample_time,
              "All Potential Names": list(set(all_names)),
              "All Potential DOBs": list(set(all_potential_dobs)),
              "All Potential Passport Numbers": list(set(all_passport_numbers)),
              "All Potential Test Sample Dates": list(set(all_potential_sample_dates)),
              "All Potential Test Sample Times": list(set(all_potential_sample_times))}
    return result
