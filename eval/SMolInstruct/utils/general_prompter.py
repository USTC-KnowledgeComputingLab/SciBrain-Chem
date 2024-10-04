TASK_TAGS = {
    'forward_synthesis': ('<SMILES>', '</SMILES>'),
    'retrosynthesis': ('<SMILES>', '</SMILES>'),
    'molecule_generation': ('<SMILES>', '</SMILES>'),
    'molecule_captioning': (None, None),
    'name_conversion-i2f': ('<MOLFORMULA>', '</MOLFORMULA>'),
    'name_conversion-i2s': ('<SMILES>', '</SMILES>'),
    'name_conversion-s2f': ('<MOLFORMULA>', '</MOLFORMULA>'),
    'name_conversion-s2i': ('<IUPAC>', '</IUPAC>'),
    'property_prediction-esol': ('<NUMBER>', '</NUMBER>'),
    'property_prediction-lipo': ('<NUMBER>', '</NUMBER>'),
    'property_prediction-bbbp': ('<BOOLEAN>', '</BOOLEAN>'),
    'property_prediction-clintox': ('<BOOLEAN>', '</BOOLEAN>'),
    'property_prediction-hiv': ('<BOOLEAN>', '</BOOLEAN>'),
    'property_prediction-sider': ('<BOOLEAN>', '</BOOLEAN>'),
}

def get_chat_content(conversation, task=None, tokenize=False, max_new_tokens=None):
    if tokenize:
        raise NotImplementedError
    available_roles = ('user', 'assistant')
    content = ''
    for idx, item in enumerate(conversation):
        role = item['role']
        assert role in available_roles, role
        if idx % 2 == 0:
            assert role == 'user'
            # content += '<s>'
            # item_content = '[INST] %s [/INST]' % item['content']
            # content += item_content
            if task == 'molecule_captioning':
                prompt_with_limited_tokens = "Below is an instruction that describes a task. Write a response not exceeding {max_new_tokens} tokens that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                item_content = prompt_with_limited_tokens.format(max_new_tokens=max_new_tokens, instruction=item['content'])
            elif task in ['property_prediction-esol', 'property_prediction-lipo']:
                add_tags = "You should utilize special tags to encapsulate special sequences, i.e., <NUMBER>...</NUMBER> for numbers."
                prompt_with_limited_tokens = "Below is an instruction that describes a task. Write a response not exceeding {max_new_tokens} tokens that appropriately completes the request. {add_tags}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                item_content = prompt_with_limited_tokens.format(max_new_tokens=max_new_tokens, add_tags=add_tags, instruction=item['content'])
            elif task in ['property_prediction-bbbp', 'property_prediction-clintox', 'property_prediction-hiv', 'property_prediction-sider']:
                add_tags = "You should utilize special tags to encapsulate your response, i.e., <BOOLEAN> Yes </BOOLEAN> for 'Yes', and <BOOLEAN> No </BOOLEAN> for 'No'."
                prompt_with_limited_tokens = "Below is an instruction that describes a task. Please respond with either 'Yes' or 'No'. {add_tags}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                item_content = prompt_with_limited_tokens.format(max_new_tokens=max_new_tokens, add_tags=add_tags, instruction=item['content'])
            else:
                add_tags = "You should utilize special tags to encapsulate special sequences, i.e., <SMILES>...</SMILES> for SMILES, <IUPAC>...</IUPAC> for IUPAC, and <MOLFORMULA>...</MOLFORMULA> for molecular formula."
                prompt_with_limited_tokens = "Below is an instruction that describes a task. Write a response not exceeding {max_new_tokens} tokens that appropriately completes the request. {add_tags}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
                item_content = prompt_with_limited_tokens.format(max_new_tokens=max_new_tokens, add_tags=add_tags, instruction=item['content'])
            content += item_content
        else:
            assert role == 'assistant'
            item_content = ' %s</s>' % item['content']
            content += item_content
    return content


class GeneralPrompter(object):

    def __init__(self, apply_chat_template_func, response_split='[/INST]'):
        self.apply_chat_template_func = apply_chat_template_func
        self.response_split = response_split

    def generate_prompt(self, chat, task=None, tokenize=False, max_new_tokens=None, *args, **kargs) -> str:
        res = self.apply_chat_template_func(chat, task=task, tokenize=tokenize, max_new_tokens=max_new_tokens, *args, **kargs)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.response_split)[-1].strip()
