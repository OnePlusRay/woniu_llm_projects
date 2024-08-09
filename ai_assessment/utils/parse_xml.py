import re
from utils.chain import get_chain

def parse_xml_output(output_str:str, tags:list, first_item_only=False):
    xml_dict = {}
    for tag in tags:
        texts = re.findall(rf'<{tag}>(.*?)</{tag}>', output_str, re.DOTALL)
        if texts:
            xml_dict[tag] = texts[0] if first_item_only else texts
    return xml_dict

def parse_xml_output_llm(output_str:str, tags:list, first_item_only=False):
    xml_dict = {}
    chain = get_chain("FULLFILL_RESULT")
    res = chain.invoke({'xml_contents': output_str, 'specific_tags': tags})
    xml_dict = parse_xml_output(res,tags,first_item_only)
    return res, xml_dict

def adjust_res_tag(output_str: str, tags:list, result_mode):
    chain = get_chain("ADJUST_RESULT_TAG")
    res = chain.invoke({'xml_contents': output_str, 'specific_tags': tags, "required_mode": result_mode})
    return res