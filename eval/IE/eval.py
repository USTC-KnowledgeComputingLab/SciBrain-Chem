import json
import pandas as pd
from typing import List
import textdistance


def readfiles(infile):

    if infile.endswith('json'):
        lines = json.load(open(infile, 'r', encoding='utf8'))
    elif infile.endswith('jsonl'):
        lines = open(infile, 'r', encoding='utf8').readlines()
        lines = [json.loads(l) for l in lines]
    else:
        raise NotImplementedError

    return lines

def levenshtein_similarity(truth: List[str], pred: List[str]) -> float:
    assert len(truth) == len(pred)
    scores = sum(textdistance.levenshtein.normalized_similarity(str(t), str(p)) for t, p in zip(truth, pred))
    return scores / len(truth)


def full_sentence_accuracy(truth: List[str], pred: List[str]) -> float:
    """Calculate the number of exact matches."""
    print(len(truth))
    print(len(pred))
    assert len(truth) == len(pred)
    correct_count = sum(int(t == p) for t, p in zip(truth, pred))
    return correct_count / len(truth)

def accuracy(columns,data_mapping):
    print(data_mapping.keys())
    score_dict = {}
    score_total = 0
    for col in columns:
        score_single = full_sentence_accuracy(data_mapping[f"ground_{col.replace(' ', '_')}"], data_mapping[f"prediction_{col.replace(' ', '_')}"])
        score_total = score_total + score_single
        score_dict[col] = score_single
    score_dict['average'] = score_total/len(columns)
    
    return score_dict


def tlevenshtein_similarity(columns, data_mapping):
    score_dict = {}
    score_total = 0
    low_score_count = 0  # 计数器初始化
    
    for col in columns:
        ground_col = f"ground_{col.replace(' ', '_')}"
        prediction_col = f"prediction_{col.replace(' ', '_')}"
        
        score_single = levenshtein_similarity(data_mapping[ground_col], data_mapping[prediction_col])
        score_total += score_single
        score_dict[col] = score_single
        
        if score_single < 0.8:
            low_score_count += 1  # 低于0.8的情况计数
    
    score_dict['average'] = score_total / len(columns)    
    return score_dict


# 以不同阈值为界的taccuracy
def tAccuracyScores(testData,predictData):
    successCase9 = 0
    for i,testdata in enumerate(testData):
        yy = testdata.keys()
        # yy.remove('battery_electrolyte')
        for key in yy:
            try:
                flag9 = 0
                score = textdistance.levenshtein.normalized_similarity(str(testdata[key]), str(predictData[i][key]))
                if score < 0.9:
                    flag9 = 1
                    break
            except:
                pass

        if flag9 == 0:
            successCase9 = successCase9 + 1

    successCase8 = 0
    for i,testdata in enumerate(testData):
        yy = testdata.keys()
        # yy.remove('battery_electrolyte')
        for key in yy:
            try:
                flag8 = 0
                score = textdistance.levenshtein.normalized_similarity(str(testdata[key]), str(predictData[i][key]))
                if score < 0.8:
                    flag8 = 1
                    break
            except:
                pass

        if flag8 == 0:
            successCase8 = successCase8 + 1
    
    return {"0.9": successCase9/len(testData),"0.8": successCase8/len(testData)}

if __name__ == "__main__":

    # accuracy & levenshtein_similarity

    testData = readfiles("")#test data path
    predictData = readfiles("")#result path
    testDataList = testData
    predictDataList = predictData

    columns = list(set().union(*testDataList))

    df_ground = pd.DataFrame(testDataList)
    df_ground = df_ground[columns]
    df_prediction = pd.DataFrame(predictDataList)
    df_ground = df_ground[columns]

    df_ground = df_ground.fillna('N/A')
    df_prediction = df_prediction.fillna('N/A')


    df_ground_column = {col: df_ground[col].tolist() for col in df_ground.columns}
    df_prediction_column = {col: df_prediction[col].tolist() for col in df_prediction.columns}

    data_mapping = {}

    for col, lst in df_ground_column.items():
        globals()[f"ground_{col.replace(' ', '_')}"] = lst
        data_mapping[f"ground_{col.replace(' ', '_')}"] = lst

    for col, lst in df_prediction_column.items():
        globals()[f"prediction_{col.replace(' ', '_')}"] = lst
        data_mapping[f"prediction_{col.replace(' ', '_')}"] = lst

    score_1 = accuracy(columns,data_mapping)

    score_2 = tlevenshtein_similarity(columns,data_mapping)

    # total accuracy
    score_t = tAccuracyScores(testData,predictData)

    output_path = " "#output path
    with open(output_path,"w",encoding='utf-8') as f:
        f.write(str(score_1) + "\n" + str(score_2) + "\n" + str(score_t))