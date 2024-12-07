from WTranscriptor.classification_utils.path_config import *
import sys

if not server_classifier:
    pass
else:
    sys.path.append(CLASSIFIER_MODULE_PATH)
    sys.path.append(NER_PATH)


    from word2number import w2n
    import random
    from Smart import Smart
    from Dull import Dull
    from Extractor import Extractor
    import re

    def clean_word(word):
        return re.sub(r'[^a-zA-Z0-9 ]', '', word)


    def join_broken_entities(data):
        i = 0
        while i < len(data) - 1:
            if data[i]['index'] + 1 == data[i+1]['index']:
                data[i]['word'] += " " + clean_word(data[i+1]['word'])
                data[i]['word'] = clean_word(data[i]['word'])
                data[i]['end'] = data[i+1]['end']
                del data[i+1]
            else:
                i += 1
        return data




    smart_classifier = Smart(base_path=CLASSIFIER_MODULE_PATH,model_path=f'files/{MODEL_NAME}/modal',
                            tokenizer_path=f'files/{MODEL_NAME}/modal',
                            mapping_dict_path = f'files/{MODEL_NAME}/modal/map.csv')
    dull_classifier = Dull(base_path=CLASSIFIER_MODULE_PATH)
    extractor = Extractor(base_path=CLASSIFIER_MODULE_PATH,
                        model_path=f'files/{NER_PATH}/',
                        mapping_dict_path=f'files/{NER_PATH}/map.csv')


    def extract_and_convert_number(text):
        words = text.split()
        max_len = 0
        final_number = None

        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                try:
                    number = w2n.word_to_num(" ".join(words[i:j]))
                    if j - i > max_len:
                        max_len = j - i
                        final_number = number
                except ValueError:
                    continue

        return final_number

    def get_entities(text,entities,verbose=False):
        print(text)
        result = extractor.predict(text,entities)
        print(result)
        filtered_list = [entry['word'] for entry in result if entry['entity'] in entities]
        text_result = ' '.join(filtered_list)
        number = extract_and_convert_number(text_result)
        if number:
            return number
        else:
            return None
    def get_random_message(msg_list):
        return msg_list[random.randint(0,len(msg_list)-1)]



    def get_top_categories_names(confidence_dict, n):
        # Sort the dictionary items by confidence in descending order
        sorted_categories = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_categories)
        # Get the names of the top categories
        top_category_names = set([category for category, _ in sorted_categories[:n]])
        
        return top_category_names

    def classify_ensemble(classification_result, verbose=False):
        intent_ai = classification_result['intent']['name']
        intent_smart = classification_result['smart_intent']
        intent_ai_confidence = classification_result['intent']['confidence']

        if intent_smart == 'already':
            if verbose:
                print('Catched Already'.center(50))
            return intent_smart

        if (intent_ai is None) and (intent_smart is None):
            if verbose:
                print('Going to Say that Again'.center(50))
            return None

        elif intent_ai_confidence <= 0.50 and (intent_smart is None):
            if verbose:
                print('Going to Say that Again'.center(50))
            return None

        elif (intent_smart is not None) and (intent_ai_confidence <= 0.50):
            if verbose:
                print('Relying on Smart'.center(50))
            return intent_smart

        elif (intent_smart is None) and (intent_ai_confidence >= 0.70):
            if verbose:
                print('Relying on Confidence'.center(50))
            return intent_ai

        else:
            if intent_ai_confidence >= 0.88:
                if verbose:
                    print('Relying on High Confidence'.center(50))
                return intent_ai
            else:
                top_candidates = get_top_categories_names(classification_result['intent_ranking'],3)
                majorty_voted = list(set(intent_smart).intersection(top_candidates))
                if len(majorty_voted)>0:
                    if verbose:
                        print('Majority Voting'.center(50))
                    return majorty_voted[0]
                    
                else:
                    if verbose:
                        print('No other way than Smart'.center(50))
                    return intent_smart

    def get_classification(transcript,verbose=False):
        # print('inside classification')
        # try:
        # print(transcript)
        result = smart_classifier.predict(transcript)
        smart_result = dull_classifier.predict(transcript)
        # print(result)
        if verbose:
            print('AI Detect',result['intent'])
            print('STR Detect',smart_result)
        result['smart_intent'] = smart_result
        # print('Result from Smart Classifier :' , smart_result)
        # final_intent = classify_ensemble(result,verbose=verbose)
        result['final_intent'] = result['intent']['name']
        return result['final_intent']
        # except Exception as e:
        #     print(f'Error {e}')


    def get_entity(transcript,verbose=False):
        # print('inside classification')
        years_ago = None
        year = None


        try:
            result = extractor.predict(transcript,ENTITY_LIST,)
            years_ago = None
            year = None
            if len(result)>0:
                iltered_list = [entry for entry in result if entry['entity'] in ENTITY_LIST]
                broken_join = join_broken_entities(iltered_list)
                # Check the entities and store them into the respective variables
                # print(broken_join)
                for item in broken_join:
                    if item['entity'] == PRORITY_LIST[0]:
                        years_ago = item['word']
                    elif item['entity'] == PRORITY_LIST[1]:
                        year = item['word']
                if year:
                    return extract_and_convert_number(year)
                elif years_ago:
                    try:
                        number = int(re.search(r'\d+', years_ago).group())
                        return number
                    except:
                        return extract_and_convert_number(years_ago)
                else:
                    return extract_and_convert_number(transcript)
            else:
                number = extract_and_convert_number(transcript)
            if number:
                return number
            else:
                return None
        except Exception as e:
            print(f'Error {e}')
            return None