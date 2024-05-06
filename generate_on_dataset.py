import os
import json
from tqdm.auto import tqdm

import fire
from datasets import load_dataset

from config import TASKS_GENERATION_SETTINGS, TASKS, DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_NEW_TOKENS
from generation import LlaSMolGeneration


def generate(
    generator: LlaSMolGeneration,
    # Data
    data_path: str = "osunlp/SMolInstruct",
    split: str = 'test',
    task: str = '',
    # Output
    output_file: str = '',
    # Running configs
    batch_size: int = 1,
    max_input_tokens: int = None,
    max_new_tokens: int = None,
    print_out=False,
    **generation_kargs,
):
    # Setting default params for certain tasks
    task_settings = TASKS_GENERATION_SETTINGS.get(task)
    if task_settings is not None:
        print('Setting configurations for %s' % task)
        for key in task_settings:
            value = task_settings[key]
            if key == 'generation_kargs':
                assert isinstance(value, dict)
                eval(key).update(value)
                print(key, '<-', value)
            else:
                if key in ('max_input_tokens', 'max_new_tokens') and eval(key) is not None:
                    pass
                else:
                    statement = '{key} = {value}'.format(key=key, value=value)
                    print(statement)
                    exec(statement)
    if max_input_tokens is None:
        max_input_tokens = DEFAULT_MAX_INPUT_TOKENS
    if max_new_tokens is None:
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    # Load dataset
    data = load_dataset(data_path, split=split, tasks=(task,))
    data = list(data)

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Check the output and continue from the break point
    mode = 'w'
    num_exist_lines = 0
    if os.path.isfile(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                num_exist_lines += 1
        if num_exist_lines > 0:
            mode = 'a'
    
    if num_exist_lines >= len(data):
        print('Already done %d / %d.' % (num_exist_lines, len(data)))
        return
    else:
        print('Todo: %d / %d' % (len(data) - num_exist_lines, len(data)))
    
    if num_exist_lines > 0:
        print('Continue with the existing %d' % num_exist_lines)

    with open(output_file, mode) as f, tqdm(total=len(data)) as pbar:
        k = num_exist_lines
        pbar.update(k)
        
        while True:
            if k >= len(data):
                break
            e = min(k + batch_size, len(data))

            batch_input = []
            
            for item in data[k: e]:
                sample_input = item['input']
                batch_input.append(sample_input)

            if len(batch_input) == 0:
                return
            
            batch_samples = data[k: e]
            
            batch_outputs = generator.generate(batch_input, batch_size=batch_size, max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens, canonicalize_smiles=False, print_out=False, **generation_kargs)

            assert len(batch_input) == len(batch_outputs)
            for sample, sample_outputs in zip(batch_samples, batch_outputs):
                if print_out:
                    tqdm.write(sample['task'])
                    tqdm.write(sample['input_text'])
                    tqdm.write(sample_outputs)
                    tqdm.write('\n')

                log = {
                    'input': sample['raw_input'], 
                    'gold': sample['raw_output'], 
                    'output': sample_outputs['output'], 
                    'task': sample['task'], 
                    'split': split, 
                    'target': sample['target'],
                    'input_text': sample_outputs['input_text'],
                    'real_input_text': sample_outputs['real_input_text'],
                }

                f.write(json.dumps(log, ensure_ascii=False) + '\n')

            pbar.update(e - k)
            k = e


def main(
    # Model
    model_name: str = "",
    base_model: str = None,
    # Data
    data_path: str = "osunlp/SMolInstruct",
    split: str = 'test',
    tasks = None,
    # Output
    output_dir: str = 'eval',
    # Running configs
    batch_size: int = 1,
    max_input_tokens: int = None,
    max_new_tokens: int = None,
    print_out=False,
    device = None,
    **generation_kargs,
):
    if tasks is None:
        tasks = TASKS
    elif isinstance(tasks, str):
        tasks = (tasks,)
    
    generator = LlaSMolGeneration(model_name=model_name, base_model=base_model, device=device)
    
    os.makedirs(output_dir, exist_ok=True)

    for task in tasks:
        generate(
            generator,
            data_path=data_path,
            split=split,
            task=task,
            output_file=os.path.join(output_dir, task + '.jsonl'),
            batch_size=batch_size,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            print_out=print_out,
            **generation_kargs
        )


if __name__ == "__main__":
    fire.Fire(main)
