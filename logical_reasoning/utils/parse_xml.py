import re

#去除thinking后解析
def parse_xml_output_without_thinking(output_str:str, tags:list, thinking_tag:str = 'thinking', first_item_only:bool = False):
    output_thinking=parse_xml_output(output_str, [thinking_tag], first_item_only)
    print('0000000',output_thinking,type(output_thinking))
    output_without_thinking = output_str
    thinking_contents = output_thinking.get(thinking_tag, "")
    if output_thinking.get(thinking_tag, ""):
        if isinstance(thinking_contents, list):
            for content in thinking_contents:
                output_without_thinking=output_str.replace(content,'').replace(f'<{thinking_tag}>','').replace(f'</{thinking_tag}>','').strip()
        elif isinstance(thinking_contents, str):
            output_without_thinking=output_str.replace(thinking_contents,'').replace(f'<{thinking_tag}>','').replace(f'</{thinking_tag}>','').strip()
    xml_dict = parse_xml_output(output_without_thinking, tags, first_item_only)
    return xml_dict

#解析xml结果
def parse_xml_output(output_str:str, tags:list, first_item_only=False):
    xml_dict = {}
    for tag in tags:
        texts = re.findall(rf'<{tag}>(.*?)</{tag}>', output_str, re.DOTALL)
        if texts:
            xml_dict[tag] = texts[0] if first_item_only else texts
    return xml_dict