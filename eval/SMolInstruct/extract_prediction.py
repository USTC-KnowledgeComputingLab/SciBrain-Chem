import os
import json

from fire import Fire

from config import TASKS, TASK_TAGS


def extract_answer_part(outputs, left_tag, right_tag, mode='tag'):
    assert mode in ('tag', 'direct')

    assert isinstance(outputs, list)
    answers = []
    for text in outputs:

        text = text.replace('<|eot_id|>', '').replace('<|end_of_text|>', '') # 删除终止符

        if mode == 'direct' or (left_tag is None and right_tag is None):
            text = text.replace('<unk>', '').replace('</s>', '').strip()
            answers.append(text.strip())
            continue

        left_tag_pos = text.find(left_tag)
        if left_tag_pos == -1:
            answers.append('')
            continue
        right_tag_pos = text.find(right_tag)
        if right_tag_pos == -1:
            answers.append('')
            continue
        text = text[left_tag_pos + len(left_tag): right_tag_pos].strip()

        # 情况太复杂了，还是借助Prompt来规范模型输出
        # text = text.split('<|eot_id|>', 1)[0] if '<|eot_id|>' in text else text # 提取终止符之前的内容
        # text = text.split('<|end_of_text|>', 1)[0] if '<|end_of_text|>' in text else text
        # text = text.split(left_tag, 1)[1] if left_tag in text else text
        # text = text.split(right_tag, 1)[0] if left_tag in text else text
        # text = text.strip()

        answers.append(text)
    return answers


def extract_prediction(output_file, prediction_file, task):
    with open(output_file, 'r') as f, open(prediction_file, 'w') as f2:
        for line in f:
            item = json.loads(line)
            outputs = item['output']
            if outputs is None:
                preds = None
            else:
                preds = extract_answer_part(outputs, *(TASK_TAGS[task]), mode='tag')
            item['pred'] = preds
            f2.write(json.dumps(item, ensure_ascii=False) + '\n')


def main(output_dir, prediction_dir, tasks=None):
    if tasks is None:
        tasks = TASKS
    elif isinstance(tasks, str):
        assert tasks in TASKS, "\"%s\" is not a valid task." % tasks
        tasks = (tasks,)
    else:
        assert isinstance(tasks, (list, tuple))
        for task in tasks:
            assert task in TASKS, "\"%s\" is not a valid task." % task
    
    assert os.path.abspath(output_dir) != os.path.abspath(prediction_dir)
    os.makedirs(prediction_dir, exist_ok=True)
    for task in tasks:
        output_file = os.path.join(output_dir, task + '.jsonl')
        if not os.path.isfile(output_file):
            print('%s: No file found. Skipped.' % task)
            continue
        prediction_file = os.path.join(prediction_dir, task + '.jsonl')
        extract_prediction(output_file, prediction_file, task)
        print('%s: Done.' % task)


if __name__ == '__main__':
    Fire(main)
