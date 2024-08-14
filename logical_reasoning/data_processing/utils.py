import re
import pdb
import json

def get_prompt(problem, question, options):
    '''构建一个逻辑推理问题的提示文本'''
    # 将选项列表转换为带有字母标签的字符串，每个选项一行，字母标签从 'A' 到 'G'，取决于有多少个选项。
    # 使用列表推导式和str.join()将转换后的选项合并成一个字符串。
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))

    prompt = f"""
你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题，最终只输出答案对应的选项字母，如\"A\"。题目如下：

### 题目:
{problem}

### 问题:
{question}
{options}
"""
    return prompt

def get_prompt_complex(problem, question, options):
    '''构建一个逻辑推理问题的提示文本（复杂版本）'''
    # 将选项列表转换为带有字母标签的字符串，每个选项一行，字母标签从 'A' 到 'G'，取决于有多少个选项。
    # 使用列表推导式和str.join()将转换后的选项合并成一个字符串。
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))

    prompt = f"""
<role>
你是一个逻辑推理专家，擅长解决闭世界假设（close-world assumption）中逻辑推理问题。
</role>

<task>
以下是一个逻辑推理的题目<problem></problem>，题目给定了所有需要的信息和事实，未观测事实都为假，所有的问题<question></question>都是基于<problem>给出，请仿照<example></example>的思考过程方法一步一步思考和分析问题，并在<options></options>选出正确答案，按照<输出要求></输出要求>中的要求，最终只输出答案，答案的格式为"A"。
</task>

<输出要求>
输出的答案只能是[A,B,C,D,E,F,G,H]中的一个。
</输出要求>

<example>
案例1
problem：在一个小镇上，有四个人分别住在不同颜色的房子里，拥有不同种类的宠物，喜欢不同的饮料，和从事不同的职业。以下是关于他们的一些线索：1.红色房子的主人是医生。2.绿色房子的主人有一只猫。3. 喜欢咖啡的人住在黄色房子里。4. 喜欢茶的人是教师。5. 白色房子的主人喜欢牛奶。根据以上线索，回答以下选择题：
question：选择题 1：哪个职业的人拥有一只猫？
options：
    A.医生 
    B.教师 
    C.律师 
    D.工程师
思考过程：
    1.根据提示，四个人的职业、房子颜色、宠物、喜欢的饮料都不一样；
    2.根据线索，住白色房子的喜欢牛奶，住黄色房子的喜欢咖啡，而教师喜欢喝茶，所以教师住的房子颜色不是白色和黄色;
    3.因为红色房子的主人是医生，所以教师也不住红色房子；
    4.根据提示，还有一间房子的颜色是绿色的，所以教师住绿色房子，绿色房子的主人养猫，所以是教师养猫。
    答案选B。

### 输出：B

案例2
problem：考虑一个简单的密码锁，它由四个旋转轮组成，每个轮上有数字0到9。密码锁的开锁密码是一个四位数。以下是一些关于密码的提示：1.第一个数字加上第二个数字的和等于7。2.第三个数字是第二个数字的两倍。3.第四个数字是第一个数字。4.第一个数字比第三个数字小2。根据以上信息，请回答以下选择题
question：选择题 2：第一个数字是多少？
options： 
    A.2 
    B.4 
    C.1 
    D.3
思考过程：
    1.根据提示2、4，第三个数字是第二个数字的两倍，第三个数字比第一个数字大2，所以第二个数字的两倍减第一个数字等于2；
    2.根据提示1，第一个数字加上的第二个数字的和等于7；
    3.因此，第二个数字的两倍减去第一个数字再加上第一个数字再加上第二个数字就等于第二个数字的三倍，等于2+7等于9，因此第二个数字为3，
    4.因此根据提示1，第一个数字为4。
    答案选B。

### 输出：B

案例3
problem：假设农夫需要将狼、鸡、和谷物从河的一边运送到另一边。农夫只能一个接一个地运送它们，每次只能携带一个（狼、鸡或谷物），并且他不能让鸡和谷物单独在一边（因为鸡会吃掉谷物），也不能让狼和鸡单独在一边（因为狼会吃掉鸡）。以下是农夫交替搭乘船运输狼、鸡和谷物的所有可能的顺序之一，请回答下列问题：
question：选择题 1：正确的运输顺序是哪一个？
options：
    A.首先带鸡过河，然后带谷物过河，接着带狼过河，再把谷物带回，带狼过河，带鸡回来，最后带鸡过河。
    B.首先带鸡过河，然后空手回来，带谷物过河，再带鸡回来，带狼过河，最后带鸡过河。
    C.首先带狼过河，然后带谷物过河，带谷物回来，带鸡过河，带狼过河，空手回来，带谷物过河。
    D.首先带狼过河，然后带鸡过河，带狼回来，带谷物过河，带鸡过河，空手回来，带狼过河。
思考过程：
    分析答案A，带鸡过河，再带谷物过河，如果回来带狼，另一边鸡就会吃掉谷物，所以A不对；
    分析答案B，带鸡过河，再带谷物过河，再把鸡带回来，带狼过河，此时河的另一边只有狼和谷物，在闭世界的假设中没有狼吃谷物，所以可以认为狼不会吃谷物，最后带鸡过河，实现了转移，所以B正确；
    分析答案C，带狼过河，此时河的一边只有鸡和谷物，根据闭世界假设，鸡会吃掉谷物，所以C不对；
    分析答案D，带狼过河，此时河的一边只有鸡和谷物，根据闭世界假设，鸡会吃掉谷物，所以D不对。
    综上所述，答案选B。

### 输出：B
</example>

### 题目:
<problem>
{problem}
</problem>

### 问题：
<question>
{question}
</question>

<options>
{options}
</options>
"""
    return prompt

def get_prompt_analysis(problem, question, options):
    '''构建一个分析逻辑推理问题的提示文本'''
    # 将选项列表转换为带有字母标签的字符串，每个选项一行，字母标签从 'A' 到 'G'，取决于有多少个选项。
    # 使用列表推导式和str.join()将转换后的选项合并成一个字符串。
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))

    prompt = f"""
你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请根据问题给出5步以内的思考思路。并给出可能的结果：

### 题目:
{problem}

### 问题:
{question}
{options}
    """
    return prompt

def get_prompt_score(problem, question, options):
    '''构建一个对结果进行评分的提示文本'''
    # 将选项列表转换为带有字母标签的字符串，每个选项一行，字母标签从 'A' 到 'G'，取决于有多少个选项。
    # 使用列表推导式和str.join()将转换后的选项合并成一个字符串。
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o}" for i, o in enumerate(options))

    prompt = f"""
你是一个逻辑推理专家，擅长解决逻辑推理问题。以下是一个逻辑推理的题目，形式为单项选择题。所有的问题都是（close-world assumption）闭世界假设，即未观测事实都为假。请逐步分析问题，最终只输出答案对应的选项字母，如"A"。题目如下：

### 题目:
{problem}

### 问题:
{question}
{options}
    """
    return prompt


def extract_from_output(input_text):
    """
    从输入文本中提取答案。

    参数:
    - input_text: 包含答案的输入文本

    返回:
    - 提取的答案字符
    """

    # 定义正则表达式模式，用于匹配答案
    ans_pattern = re.compile(r"答案是：(.)", re.S)

    # 使用正则表达式查找所有匹配的答案
    problems = ans_pattern.findall(input_text)

    # 返回第一个匹配的答案
    return problems[0]


def get_prompt_improved(problem, question, options):
    '''构建一个逻辑推理问题的提示文本'''
    options = '\n'.join(f"{'ABCDEFG'[i]}. {o.strip()}" for i, o in enumerate(options))

    prompt = f"""
你是一个逻辑推理专家，擅长解决逻辑推理问题。请根据以下题目的信息逐步分析并选择正确的选项。所有的问题都是闭世界假设，即未观测事实都为假。请注意，最终只输出选项的字母，如'A'。

### 题目:
{problem.strip()}

### 问题:
{question.strip()}

### 选项:
{options}

### 示例分析:
题目是："有苹果和鸡肉是被公认的食物。除此之外，如果有人喜欢吃某样东西，且这个人不会因为吃这个东西而死亡，则这样东西也被认为是食物。比尔喜欢吃花生，且吃花生不会使他死亡。约翰只喜欢食物。"

问题是："约翰喜不喜欢花生？"

选项是：
A. 是
B. 否

**分析过程:** 根据比尔的情况，花生被认为是食物，因此约翰应该喜欢花生。

在这种情况下，正确答案是：A。因此最终输出的内容是：'A'。

请分析上述信息并给出你的答案。
    """
    return prompt

def split_dataset(input_file, output_file):
    """
    将数据集中的每个子问题作为一个独立的问题，拆分数据集并保存到新的文件中。

    参数:
    - input_file: 输入文件路径
    - output_file: 输出文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        new_id = 1000
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                item = json.loads(line)
                problem = item['problem']
                for question in item['questions']:
                    new_item = {
                        'problem': problem,
                        'question': question['question'],
                        'options': question['options'],
                        'answer': question.get('answer', None),  # 如果有答案则保留
                        # 'gpt4':question.get('gpt4', None),
                        'id': f'round1_train_data_{new_id}'
                    }
                    outfile.write(json.dumps(new_item, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {line}")
                print(e)
            new_id += 1
    return output_file