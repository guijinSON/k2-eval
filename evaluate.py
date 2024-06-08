import argparse
import pandas as pd
from constraints import (
    no_commas, count_sentence, count_pos_under, count_paragraph, count_repeat, count_pos_non, count_pos_over,
    check_braced_strings, honorific_haeyo, honorific_hao, check_first_consonant, 
    check_middle_vowel, check_final_consonant
)

def calculate_accuracy(data, constraint_name, check_function):
    relevant_data = data[data['constraint'] == constraint_name]['regeneration']
    if len(relevant_data) == 0:
        return 0  # Prevent division by zero
    return sum(1 for item in relevant_data if check_function(item)) / len(relevant_data)

def calculate_different_accuracy(data, constraint_name, check_function, decrease_rate=0.1):
    before_data = data[data['constraint'] == constraint_name]['generation'].apply(check_function)
    after_data = data[data['constraint'] == constraint_name]['regeneration'].apply(check_function)
    
    if len(before_data) == 0 or len(after_data) == 0:
        return 0  # Prevent division by zero


    accuracy = sum(1 for before, after in zip(before_data, after_data) if before > after and (before - after) / before >= decrease_rate) / len(before_data)
    
    return accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Process model path for text generation analysis.")
    parser.add_argument("model_path", type=str, help="Specify the model path for the input and output CSV files.")
    return parser.parse_args()

def main(model_path):
    gen = pd.read_csv(f'results/{model_path}_gen.csv')
    kno = pd.read_csv(f'results/{model_path}_kno.csv')
    con = pd.read_csv(f'results/{model_path}_con.csv')

    results = {
        'kno_accuracy': sum(['A', 'B', 'C', 'D'][int(g)] == p for g, p in kno[['gold_answer', 'predict']].values) / len(kno),
        'no_commas': calculate_accuracy(con, 'no_commas', no_commas),
        
        'tot_sentence_5': calculate_accuracy(con, 'max_sentences_3', lambda gen: count_sentence(gen,5)),
        'tot_sentence_10': calculate_accuracy(con, 'max_sentences_4', lambda gen: count_sentence(gen,10)),
        'tot_sentence_15': calculate_accuracy(con, 'max_sentences_5', lambda gen: count_sentence(gen,15)),
        
        'max_verbs_3': calculate_accuracy(con, 'max_verbs_3', lambda gen: count_pos_under(gen, '동사',3)),
        'max_verbs_5': calculate_accuracy(con, 'max_verbs_5', lambda gen: count_pos_under(gen, '동사',5)),
        'max_verbs_7': calculate_accuracy(con, 'max_verbs_7', lambda gen: count_pos_under(gen, '동사',7)),
        'max_verbs_9': calculate_accuracy(con, 'max_verbs_9', lambda gen: count_pos_under(gen, '동사',9)),

        'max_adjectives_3': calculate_accuracy(con, 'max_adjectives_3', lambda gen: count_pos_under(gen, '형용사',3)),
        'max_adjectives_5': calculate_accuracy(con, 'max_adjectives_5', lambda gen: count_pos_under(gen, '형용사',5)),
        'max_adjectives_7': calculate_accuracy(con, 'max_adjectives_7', lambda gen: count_pos_under(gen, '형용사',7)),
        'max_adjectives_9': calculate_accuracy(con, 'max_adjectives_9', lambda gen: count_pos_under(gen, '형용사',9)),
        
        'max_paragraphs_2': calculate_accuracy(con, 'max_paragraphs_2', lambda gen: count_paragraph(gen,2)),
        'max_paragraphs_3': calculate_accuracy(con, 'max_paragraphs_3', lambda gen: count_paragraph(gen,3)),
        'max_paragraphs_4': calculate_accuracy(con, 'max_paragraphs_4', lambda gen: count_paragraph(gen,4)),
        
        'no_repeated_nouns': calculate_accuracy(con, 'no_repeated_nouns', count_repeat),
        'no_pronouns': calculate_accuracy(con, 'no_pronouns', lambda gen: count_pos_non(gen, '대명사')),
        'no_dependent_nouns': calculate_accuracy(con, 'no_dependent_nouns', lambda gen: count_pos_non(gen, '의존 명사')),
        'no_conjunctive_adverbs': calculate_accuracy(con, 'no_conjunctive_adverbs', lambda gen: count_pos_non(gen, '접속 부사')),
        'bracket_proper_nouns': calculate_accuracy(con, 'bracket_proper_nouns', check_braced_strings),
        'least_attributives': calculate_accuracy(con, 'least_attributives', lambda gen: count_pos_over(gen, '관형사')),
        'honorific_haeyo': calculate_accuracy(con, 'honorific_haeyo', honorific_haeyo),
        'honorific_hao': calculate_accuracy(con, 'honorific_hao', honorific_hao),
        'first_consonant': calculate_different_accuracy(con, 'first_consonant', check_first_consonant),
        'middle_vowel': calculate_different_accuracy(con, 'middle_vowel', check_middle_vowel),
        'final_consonant': calculate_different_accuracy(con, 'final_consonant', check_final_consonant)
    }

    return results

if __name__ == '__main__':
    args = parse_args()
    model_path = args.model_path#.split('/')[1]
    accuracies = main(model_path)
    pd.DataFrame(accuracies,index=[0]).to_csv(f'results/{model_path}-results.csv')