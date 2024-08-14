import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import json
import random

# 定义增强器
synonym_aug = naw.SynonymAug(aug_src='wordnet')
random_insert_aug = naw.ContextualWordEmbsAug(action="insert")
random_delete_aug = naw.RandomWordAug(action="delete")
random_swap_aug = naw.RandomWordAug(action="swap")
back_translation_aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en')
char_aug = nac.RandomCharAug(action="insert")

# 原始数据
data = {
    "instruction": "你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。请逐步分析问题，最终只输出答案的选项字母，如\"A\"。题目如下：\n\n### 题目:\n有一个英文到法文的词汇表，包含以下对应词汇：\n\n1. the -> le\n2. cat -> chat\n3. jumps -> sauts\n4. over -> sur\n5. moon -> lune\n6. cow -> vache\n7. plays -> jouer\n8. fiddle -> violon\n9. egg -> bougre\n10. falls -> des chutes\n11. off -> de\n12. wall -> mur\n\n根据这个词汇表，翻译以下英文句子成法文：\n\n### 问题:\n选择题 1：\n英文句子 \"the cat jumps over the moon\" 翻译成法文是：\nA. le chat saute sur la lune\nB. le chat sauts sur le lune\nC. le sauts chat sur le lune\nD. le chat sauts sur le lune\n",
    "input": "",
    "output": "D"
}

# 数据增强函数
def augment_data(data, augmenters, num_augments=5):
    augmented_data = []
    for _ in range(num_augments):
        new_data = data.copy()
        augmenter = random.choice(augmenters)
        # 对instruction部分进行增强
        new_data["instruction"] = augmenter.augment(data["instruction"])
        augmented_data.append(new_data)
    return augmented_data

# 增强器列表
augmenters = [synonym_aug, random_insert_aug, random_delete_aug, random_swap_aug, back_translation_aug, char_aug]

# 增强数据
augmented_data = augment_data(data, augmenters)

# 打印增强后的数据
for i, aug_data in enumerate(augmented_data):
    print(f"增强后的数据 {i+1}:")
    print(json.dumps(aug_data, ensure_ascii=False, indent=2))

