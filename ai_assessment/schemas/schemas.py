from pydantic import BaseModel, ValidationError, validator
from enum import Enum
from typing import Any


class JobRequest(BaseModel):
    jobId: str
    tag: str
    data: Any

tag_2_result_mapping = {
    "ANNUAL_REPORT_QUALITY": ['差', '中', '好'],
    "POTENTIAL_CHECK": ['意向客户', '非意向客户'],
    "INITIAL_CALL": ['好', '差'],
    "POLICY_ANNUAL": ['符合', '不符合']
}

# 定义数据模型
class ParseResult(BaseModel):
    parse_thinking: str
    result_mode: list
    parse_result: str

    @validator('parse_result', always=True)
    def check_result_form(cls, value, values):
        result = value
        result_mode = values.get('result_mode')
        if result in result_mode:
            return result
        else:
            raise ValueError('result不符合规范')