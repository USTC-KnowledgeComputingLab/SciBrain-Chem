import json
import os
import re

      
# def parse(s):
#       match = re.match(r"Question \d+: (.+)", s)
#       if match:
#        return match.group(1)
#       return None

### Fill the blank
testcase = 1
model_pred_path = "" # Output file path of main_new.py
result_dir = "" # The final results
### ==============

answers = []
test_path = "./data/" + str(testcase) + ".jsonl"
with open(test_path, 'r') as f:

        for line in f:

            item = json.loads(line.strip())

            try:
                answers.append(item["answer"])
            except:
                answers.append(item["Answer"])

output = []
with open(model_pred_path, 'r') as o:
      for l in o:
            item = json.loads(l)
            for key in item:
                  output.append(item[key])
#print(output)

correct = 0
for i in range(len(answers)):
      res = output[i]
      match = re.search(r'The correct answer is (.+)', res)

      if match:
            res = match.group(1)  # 提取匹配的内容
      # res = parse(output[i])
      # print(res)
      if str(answers[i]) == str(res[0]) or str(answers[i]) == str(res[-1]) :
            correct +=1
            print("correct!")
      else:
            with open(result_dir,"a") as result:
                  result.write(f"Question{i+1} is wrong. Correct answer is {answers[i]} but prediction is {output[i]}.")
                  result.write("\n")
            print(f"Question{i+1} is wrong. Correct answer is {answers[i]} but prediction is {output[i]}.")

with open(result_dir,"a") as result:
      print(f"accuracy is {correct} / {len(answers)} = {correct/len(answers)}.")
      result.write(f"accuracy is {correct} / {len(answers)} = {correct/len(answers)}.")