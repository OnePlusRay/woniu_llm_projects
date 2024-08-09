from pydantic import BaseModel, ValidationError, validator
from enum import Enum
from typing import Any

# 定义请求体模型
class RequestModel(BaseModel):
    id: str
    uploadUrl: str
    callbackUrl: str

# 定义响应体模型
class ResponseModel(BaseModel):
    id: str
    downloadUrl: str
    processResult: bool