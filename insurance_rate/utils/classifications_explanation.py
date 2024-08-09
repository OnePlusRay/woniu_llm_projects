import json

#获取各类别及对应的解释
def get_classifications_explanation():
    with open(r'./assets/classifications_explanation.json', 'r', encoding='utf-8') as f:
        classifications_explanation = json.load(f)
    classifications = list(classifications_explanation.keys())
    explanations = ''
    for index, classification in enumerate(classifications_explanation):
        if classifications_explanation[classification]:
            explanations += f'{index+1}、{classification}：{classifications_explanation[classification]}\n'
        else:
            explanations += f'{index+1}、{classification}\n'
    return classifications, explanations

if __name__ == "__main__":
    print(get_classifications_explanation())