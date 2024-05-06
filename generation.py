import torch
from transformers import GenerationConfig

from utils.chat_generation import generate_chat
from utils.general_prompter import GeneralPrompter, get_chat_content
from utils.smiles_canonicalization import canonicalize_molecule_smiles

from model import load_tokenizer_and_model


def tokenize(tokenizer, prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    result = tokenizer(
        prompt,
        truncation=False,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result


def canonicalize_smiles_in_text(text, tags=('<SMILES>', '</SMILES>'), keep_text_unchanged_if_no_tags=True, keep_text_unchanged_if_error=False):
    try:
        left_tag, right_tag = tags
        assert left_tag is not None
        assert right_tag is not None
        
        left_tag_pos = text.find(left_tag)
        right_tag_pos = None
        if left_tag_pos == -1:
            assert right_tag not in text, 'The input text "%s" only contains the right tag "%s" but no left tag"%s"' % (text, right_tag, left_tag)
            return text
        else:
            right_tag_pos = text.find(right_tag)
            assert right_tag_pos is not None, 'The input text "%s" only contains the left tag "%s" but no right tag"%s"' % (text, left_tag, right_tag)
    except AssertionError:
        if keep_text_unchanged_if_no_tags:
            return text
        raise
    
    smiles = text[left_tag_pos + len(left_tag) : right_tag_pos].strip()
    try:
        smiles = canonicalize_molecule_smiles(smiles, return_none_for_error=False)
    except KeyboardInterrupt:
        raise
    except:
        if keep_text_unchanged_if_error:
            return text
        raise

    new_text = text[:left_tag_pos] + ('' if (left_tag_pos == 0 or text[left_tag_pos - 1] == ' ') else ' ') + left_tag + ' ' + smiles + ' ' + right_tag + ' ' + text[right_tag_pos + len(right_tag):].lstrip()
    return new_text


class LlaSMolGeneration(object):
    def __init__(self, model_name, base_model=None, device=None):
        self.prompter = GeneralPrompter(get_chat_content)

        self.tokenizer, self.model = load_tokenizer_and_model(model_name, base_model=base_model, device=device)
        self.device = self.model.device  # TODO: check if this can work

    def create_sample(self, text, canonicalize_smiles=True, max_input_tokens=None):
        if canonicalize_smiles:
            real_text = canonicalize_smiles_in_text(text)
        else:
            real_text = text
        
        sample = {'input_text': text}
        chat = generate_chat(real_text, output_text=None)
        full_prompt = self.prompter.generate_prompt(chat)
        sample['real_input_text'] = full_prompt
        tokenized_full_prompt = tokenize(self.tokenizer, full_prompt, add_eos_token=False)
        sample.update(tokenized_full_prompt)
        if max_input_tokens is not None and len(tokenized_full_prompt['input_ids']) > max_input_tokens:
            sample['input_too_long'] = True
        
        return sample
    
    def _generate(self, input_ids, max_new_tokens=1024, **generation_settings):
        generation_config = GenerationConfig(
            pad_token_id=self.model.config.pad_token_id,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            **generation_settings,
        )
        self.model.eval()
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = self.tokenizer.batch_decode(s, skip_special_tokens=False)
        output_text = []
        for output_item in output:
            text = self.prompter.get_response(output_item)
            output_text.append(text)

        return output_text, output

    def generate(self, input_text, batch_size=1, max_input_tokens=512, max_new_tokens=1024, canonicalize_smiles=True, print_out=False, **generation_settings):
        if isinstance(input_text, str):
            input_text = [input_text]
        else:
            input_text = list(input_text)
        assert len(input_text) > 0

        samples = []
        for text in input_text:
            sample = self.create_sample(text, canonicalize_smiles=canonicalize_smiles, max_input_tokens=max_input_tokens)
            samples.append(sample)
        
        all_outputs = []
        k = 0
        while True:
            if k >= len(samples):
                break
            e = min(k + batch_size, len(samples))

            batch_samples = []
            skipped_samples = []
            batch_outputs = []
            original_index = {}
            
            for bidx, sample in enumerate(samples[k: e]):
                if 'input_too_long' in sample and sample['input_too_long']:
                    original_index[bidx] = ('s', len(skipped_samples))
                    skipped_samples.append(sample)
                    continue
                original_index[bidx] = ('b', len(batch_samples))
                batch_samples.append(sample)

            if len(batch_samples) > 0:
                input_ids = {'input_ids': [sample['input_ids'] for sample in batch_samples]}
                input_ids = self.tokenizer.pad(
                    input_ids,
                    padding=True,
                    return_tensors='pt'
                )
                input_ids = input_ids['input_ids'].to(self.device)
                batch_output_text, _ = self._generate(input_ids, max_new_tokens=max_new_tokens, **generation_settings)
                num_batch_samples = len(batch_samples)
                ko = 0
                num_return_sequences = 1 if 'num_return_sequences' not in generation_settings else generation_settings['num_return_sequences']
                for sample in range(num_batch_samples):
                    sample_outputs = []
                    for _ in range(num_return_sequences):
                        sample_outputs.append(batch_output_text[ko])
                        ko += 1
                    batch_outputs.append(sample_outputs)
                
            new_batch_samples = []
            new_batch_outputs = []

            for bidx in sorted(original_index.keys()):
                place, widx = original_index[bidx]
                if place == 'b':
                    sample = batch_samples[widx]
                    output = batch_outputs[widx]
                elif place == 's':
                    sample = skipped_samples[widx]
                    output = None
                else:
                    raise ValueError(place)
                new_batch_samples.append(sample)
                new_batch_outputs.append(output)

            batch_samples = new_batch_samples
            batch_outputs = new_batch_outputs

            assert len(batch_samples) == len(batch_outputs)
            for sample, sample_outputs in zip(batch_samples, batch_outputs):
                if print_out:
                    print('=============')
                    print('Input: %s' % sample['input_text'])
                    if sample_outputs is None:
                        print('Output: None (Because the input text exceeds the token limit (%d) )' % max_input_tokens)
                    else:
                        for idx, output_text in enumerate(sample_outputs, start=1):
                            print('Output %d: %s' % (idx, output_text))
                    print('\n')

                log = {
                    'input_text': sample['input_text'], 
                    'real_input_text': sample['real_input_text'],
                    'output': sample_outputs,
                }

                all_outputs.append(log)

            k = e
        
        return all_outputs
