import os
import json
import fire
from datasets import load_dataset

from config import TASKS, TASKS_WITH_SEMICOLON_REPLACE, TASKS_WITH_READING_GOLD_FROM_DATASET
from utils.metrics import calculate_smiles_metrics, calculate_formula_metrics, calculate_text_metrics, calculate_number_metrics, calculate_boolean_metrics


def read_result(prediction_dir, task, replace_semicolon=False, read_gold_from_dataset=False):
    input_to_gold = None
    if read_gold_from_dataset:
        split_set = load_dataset('osunlp/SMolInstruct', tasks=(task,), split='test')
        input_to_gold = dict()
        for sample in split_set:
            input_key = sample['raw_input']
            if 'target' in sample and sample['target'] is not None:
                input_key = (input_key, sample['target'])
            gold_answer = sample['raw_output']
            if input_key not in input_to_gold:
                input_to_gold[input_key] = []
            input_to_gold[input_key].append(gold_answer)
        
        for input_key in input_to_gold:
            input_to_gold[input_key] = set(input_to_gold[input_key])


    pred_list = []
    gold_list = []

    file_path = os.path.join(prediction_dir, task + '.jsonl')
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            item = json.loads(line)
            if read_gold_from_dataset:
                input_key = item['input']
                if 'target' in item and item['target'] is not None:
                    input_key = (input_key, item['target'])
                golds = input_to_gold[input_key]
                assert item['gold'] in golds
                if replace_semicolon:
                    new_golds = []
                    for one_gold in golds:
                        one_gold = one_gold.replace(';', '.')
                        new_golds.append(one_gold)
                    golds = new_golds
                else:
                    golds = list(golds)
            else:
                gold = item['gold']
                if replace_semicolon:
                    gold = gold.replace(';', '.')
                golds = [gold]
            gold_list.append(golds)

            preds = item['pred']
            if preds is None:  # Input too long, so skipped this sample
                pred_list.append(preds)
                continue

            new_preds = []
            for pred in preds:
                if replace_semicolon and pred is not None:
                    pred = pred.replace(';', '.')
                new_preds.append(pred)
            pred_list.append(new_preds)
    return pred_list, gold_list


def main(prediction_dir, tasks=TASKS):
    for task in tasks:
        print('===== %s =====' % task)
        if not os.path.isfile(os.path.join(prediction_dir, task + '.jsonl')):
            print('No file found. Skipped.\n')
            continue

        replace_semicolon = True if task in TASKS_WITH_SEMICOLON_REPLACE else False
        pred_list, gold_list = read_result(
            prediction_dir, task, 
            replace_semicolon=replace_semicolon, 
            read_gold_from_dataset=True if task in TASKS_WITH_READING_GOLD_FROM_DATASET else False,
        )
        print('Altogether %d samples.' % len(pred_list))

        if len(pred_list) == 0:
            print()
            continue

        if task in ('forward_synthesis', 'molecule_generation', 'name_conversion-i2s'):
            r = calculate_smiles_metrics(pred_list, gold_list)
        elif task in ('retrosynthesis',):
            r = calculate_smiles_metrics(pred_list, gold_list, metrics=('exact_match', 'fingerprint', 'multiple_match'))
        elif task in ('molecule_captioning',):
            r = calculate_text_metrics(pred_list, gold_list)
        elif task in ('name_conversion-i2f', 'name_conversion-s2f'):
            r = calculate_formula_metrics(pred_list, gold_list, metrics=('element_match',))
        elif task in ('name_conversion-s2i',):
            r = calculate_formula_metrics(pred_list, gold_list, metrics=('split_match',))
        elif task in ('property_prediction-esol', 'property_prediction-lipo'):
            r = calculate_number_metrics(pred_list, gold_list)
        elif task in ('property_prediction-bbbp', 'property_prediction-clintox', 'property_prediction-hiv', 'property_prediction-sider'):
            r = calculate_boolean_metrics(pred_list, gold_list)
        else:
            raise ValueError(task)

        for key in r:
            print('%s:\t' % key, r[key])
        print()


if __name__ == '__main__':
    fire.Fire(main)
