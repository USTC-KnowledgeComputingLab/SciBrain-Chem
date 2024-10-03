from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import SciQDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

sciq_reader_cfg = dict(
    # input_columns=['support', 'question', 'distractor1', 'distractor2', 'distractor3', 'correct_answer'],
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer',
)

_hint = f'There is a multiple choice question about science. Answer the question by replying A, B, C or D.'

human_prompt=f'{_hint}\nQuestion: {{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '

sciq_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(role='HUMAN', prompt=human_prompt)
        ])
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512)
)

sciq_eval_cfg = dict(
    # evaluator=dict(type=AccEvaluator),
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

sciq_datasets = [
    dict(
        abbr= f'sciq',
        type=SciQDataset,
        path='./data/sciq',
        reader_cfg=sciq_reader_cfg,
        infer_cfg=sciq_infer_cfg,
        eval_cfg=sciq_eval_cfg
    )
]
