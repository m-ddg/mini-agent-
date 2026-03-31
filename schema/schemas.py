from pydantic import BaseModel, Field
from typing import Optional


class function_call(BaseModel):
    function_name: str
    args: Optional[BaseModel]


class tool_call(BaseModel):
    tool_call: str
    type: str   # 目前仅有调用函数 type = function
    function: function_call

class Message(BaseModel):
    role: str # system, user, assistant, tool
    content: str
    thinking: Optional[str] = None
    tool_calls: Optional[list[tool_call]] = None
