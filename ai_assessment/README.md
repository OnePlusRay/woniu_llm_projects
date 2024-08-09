# ai_assessment

## 项目简介
这是一个基于AI每日定时进行保单评价，子项目包括：
- [保单年检报告分析](#保单年检报告分析)
- [意向客户判断](#意向客户判断)
- [通话记录总结](#通话记录总结)
- [新客首播通话记录](#新客首播通话记录)
- [通话记录每日评价](#通话记录每日评价)
- [通话记录每月评价](#通话记录每月评价)
- [保单梳理通话评价](#保单梳理通话评价)
- [通话违规检查](#通话违规检查)


## 流程
启动api接口，调用后端接口，后端调用ai模型，输入请求体job_request，通过输入的参数tag确定要执行的项目，并执行任务，会在后台增加任务，完成后发给mq。

## 文件结构
- schemas/：输入和输出数据结构
  - ```schemas.py```：定义 api 的请求体模型和响应体模型（响应体模型似乎没用到）
    ```python
    class JobRequest(BaseModel):
        jobId: str
        tag: str
        data: Any
    ```
- utils/：api 调用链相关函数
  - ```chain.py```：通过 LangChain 构建 prompt-llm-output 调用链
  - ```llm.py```：可选的大模型 API 函数
  - ```llm_with_sql.py```：重写的 llm，包含 sql 存取功能
  - ```parse_xml.py```：用于解析和处理大模型输出
  - ```conversation_processing.py```：用于数据预处理，将一系列对话行拼接成一个完整的对话字符串
  - ```custom_chat_openai.py```：自定义 langchain 配置（一般情况不需要修改）
- routers/: 存放tag所对应的每个子项目的运行函数，每个文件对应一个路由。
    -_init_.py: 路由初始化，将每个子项目的运行函数注册到路由中。
- middlewares/：中间件
  - ```cors.py```：添加 CORS（跨源资源共享）中间件，CORS 中间件允许你的 API 能够被不同的域名访问
- assets/：存放各子项目的提示词的地方
- ```.env```：环境路径配置，主要包括以下内容
  - GPT-4 和 GPT-3.5-turbo 的 api-key 路径配置
  - claude-3 的 api-key 路径配置
  - llama-3-70b 的 api-key 路径配置
  - qwen-1.5-[7b, 14b, 110b] 以及 qwen-2-72b 的 api-key 路径配置
  - mysql 数据库环境配置（这里是本地环境）
  - rocketmq 路径配置
- ```requirements.txt```：项目依赖
- ```start.sh```: 用于配置 proxychains，生成按日期命名的日志文件，并通过 proxychains 启动一个 FastAPI 应用。
- ```local_start.sh```：用于通过 proxychains 工具启动一个 FastAPI 应用，并在终端启动。
- ```main.py```：FastAPI 应用实例，定义了路由和后台任务。


## 启动
- 先做好配置
    ```
    cd ai-assessment
    pip install -r requirements.txt
    ```
- 在终端输入：
    ```
    chmod +x local_start.sh
    ./local_start.sh
    ```
>启动api接口，通过local_start.sh上的信息确定主机和端口为：10.0.0.25:9000，并指向fastapi的应用实例的路径：main.py的@app.post("/process/")，所以fastapi的应用实例的路径为：http://10.0.0.25:9000/process/

## 调用
- 调用接口，通过输入的参数tag确定要执行的项目，并执行任务。
- 在foxapi主体中输入参数：job_request：
    ```json
        {
            "jobId":"111",
            "tag":"标签",
            "data":{...}
        }
    ```
- 参数传入main.py中的@app.post("/process/")里的函数process_inputs()，并通过代码：
    ```python
        process_func = tag_2_func_mapping[job_request.tag]
    ```
- 找到对应项目的函数，并记录为process_func，然后通过：
    ```python
        background_tasks.add_task(process_func, job_request)
    ```
- 执行process_func(job_request)，执行任务。
> background_tasks可以将process_func和job_request作为参数添加到后台任务中，以便在响应发送给客户端之后执行这个任务。

## 任务执行
### 保单年检报告分析
- 保单年检报告分析的tag为`"ANNUAL_REPORT_QUALITY"`。
- `"ANNUAL_REPORT_QUALITY"`是通过分析客户和经纪人之间的的电话通话记录和微信聊天记录，根据对话中经纪人是否为客户提供了客户个性化信息分析进行评价，评价分为差、中、好三个等级。
    - `data `客户和经纪人的客户个性化信息。

>- 当输入参数tag为`"ANNUAL_REPORT_QUALITY"`时，会通过`process_func(job_request)`调用函数`annual_report_quality_process_func(job_request)`，执行任务。
>- 首先会从`get_chain(tag)`中获取`"ANNUAL_REPORT_QUALITY"`对应的提示词：`annual_report_quality_process_func.txt`，并且连线使用的ai模型，返回chain。
>- 用`chain.invoke({"content": job_request.data})`将处理过的对话数据传入到提示词的`{content}`位置，分析客户数据，并给出评价。

### 意向客户判断
- 意向客户判断的tag为`"POTENTIAL_CHECK"`
- `"POTENTIAL_CHECK"`是通过分析客户和经纪人之间的的电话通话记录和微信聊天记录，总结对话中客户回复的积极性和认可度，判断客户是否是潜在的意向客户，并最终给出[意向客户，非意向客户]的结论。
    - `data `客户和经纪人的通话记录和微信聊天记录。
    ```json
        {
            "jobId": "",
            "tag": "POTENTIAL_CHECK",
            "data":  [
                "callContent": [{"speaker": "A", "message": "内容"},{"speaker": "B", "message": "内容"}],
                "wechatContent": [{"speaker": "A", "message": "内容"},{"speaker": "B", "message": "内容"},]
                ]
        }
    ``` 

>- 当输入参数tag为`"POTENTIAL_CHECK"`时，会通过`process_func(job_request)`调用函数`potential_client_check_process_func(job_request)`，执行任务。
>- 首先会从`get_chain(tag)`中获取`"POTENTIAL_CHECK"`对应的提示词：`potential_client_check_process_func.txt`，并且连线使用的ai模型，返回chain。
>- 用`chain.invoke({"content": conversation})`将处理过的对话数据传入到提示词的`{content}`位置，并用`invoke`获取对客户和经纪人的通话内容和微信聊天内容总结后客户是否是意向用户。
>- 最后进一步处理输出的评价，只提取\<thinking>和\<result>。



### 通话记录总结
- 通话记录总结的tag为`"SUMMARY"`。
- `"SUMMARY"` 的功能是将客户和经纪人的语音通话记录内容进行总结和提炼。
    - `data `客户和经纪人的通话记录
    ```json
        {
            "jobId": "",
            "tag": "SUMMARY",
            "data": [
                {"time": "时长", "role": "角色1", "word":"内容"},
                {"time": "时长", "role": "角色2", "word":"内容"}
                ]
        }
    ```        
>- 当输入参数tag为`"SUMMARY"`时，会通过`process_func(job_request)`调用函数`call_record_summary_process_func(job_request)`，执行任务。
>- 首先会从`get_chain(tag)`中获取`"SUMMARY"`对应的提示词：`call_record_summary_process_func.txt`，并且连线使用的ai模型，返回chain。
>- 用`chain.invoke({"content": conversation})`将处理过的对话数据传入到提示词的`{content}`位置，并用`invoke`获取对客户和经纪人的通话内容的总结。

### 新客首播通话记录
- 通话记录每日评价的tag为`"INITIAL_CALL"`。

- `INITIAL_CALL`是分析经纪人和新客的首次通话记录的质量，根据提示词中的提供的规则进行评价，由ai生成哪些要求达到了哪些没做到，并生成结果。
    - data的主要内容有通话数据，主要结构如下。
        ```json
            {
                "jobId": "",
                "tag": "POLICY_ANNUAL",
                "data": [
                    {"time": "时长", "role": "角色1", "word":"内容"},
                    {"time": "时长", "role": "角色2", "word":"内容"}
                    ]
            }
        ```

>- 当输入参数tag为`"INITIAL_CALL"`时，会通过`process_func(job_request)`调用函数`initial_call_quality_process_func(job_request)`，执行任务。
>- 首先会从`get_chain(tag)`中获取`INITIAL_CALL`对应的提示词：`initial_call_quality_prompt.txt`，并且连线使用的ai模型，返回chain。
>- 用`chain.invoke({"content": conversation})`将处理过的对话数据传入到提示词的`{content}`位置，并用`invoke`获取ai的评价。
>- 最后进一步处理输出的评价，只提取\<thinking>和\<result>。

### 通话记录每日评价
- 通话记录每日评价的tag为`"DAILY_SERVICE_SUMMARY"`。

- `DAILY_SERVICE_SUMMARY`是通过分析客户和经纪人之间的的沟通内容，总结对话主题和处理了哪些问题，分析是否需要进一步沟通。
    - data的主要内容有电话聊天数据和微信聊天记录，主要结构如下。
        ```json
            {
                "jobId": "",
                "tag": "DAILY_SERVICE_SUMMARY",
                "data": {
                    "callContentList": [{"callContent": [], "servcieProject": ""}],
                    "wechatContentList": [{"wechatContent": [],"servcieProject": ""}]
                }
            }
        ```

>- 当输入参数tag为`"DAILY_SERVICE_SUMMARY"`时，会通过`process_func(job_request)`调用函数`daily_potential_client_check_process_func(job_request)`，执行任务。
>- 首先会从`get_chain(tag)`中获取`DAILY_SERVICE_SUMMARY`对应的提示词：`daily_service_summary_prompt.txt`，并且连线使用的ai模型，返回chain。
>- 用`chain.invoke({"content": conversation})`将处理过的对话数据传入到提示词的`{content}`位置，并用`invoke`获取ai的评价。

### 通话记录每月评价
- 通话记录每月评价的tag为`"MONTH_SERVICE_SUMMARY"`。

- `MONTH_SERVICE_SUMMARY`是通过分析客户和经纪人之间的的一个月的对话总结（该总结基于[通话记录每日评价](通话记录每月评价)），总结过去一个月都沟通了哪些问题，交易成功的保险记录，和被投诉的记录。
    - `data`的主要内容有聊天记录总结、交易成功的保险记录、被投诉的记录，具体组成如下：
        ```json
        {
            "jobId": "",
            "tag": "MONTH_SERVICE_SUMMARY",
            "data": {
                "serviceSummarys":[{"date":"yyyy-MM-dd","summary":""}],
                "policyRecords" : [{"productName" : "","productType" : "","fee" : 000}],
                "complaintRecords" : [{"content" : "","time" : "yyyy-MM-dd HH:mm:ss"}]
            }
        }
        ```

>- 当输入参数tag为`"MONTH_SERVICE_SUMMARY"`时，会通过`process_func(job_request)`调用函数`month_potential_client_check_process_func(job_request)`，执行任务。
>- 首先会从`get_chain(tag)`中获取`MONTH_SERVICE_SUMMARY`对应的提示词：`month_service_summary_prompt.txt`，并且连线使用的ai模型，返回chain。
>- 用`chain.invoke({"content": conversation})`将处理过的对话数据传入到提示词的`{content}`位置，并用invoke获取ai的评价。

### 保单梳理通话评价
- 通话流程质量检验的tag为`"POLICY_ANNUAL"`。

- `POLICY_ANNUAL`是通过分析客户和经纪人之间的的沟通内容，结合提示词中给到的工作流程和工作要求，分析经纪人在对话中是否有安装工作流程和工作要求为顾客答疑解惑和梳理保单，并最终给出[符合,不符合]的结论。
    - data的主要内容有电话聊天数据。
        ```json
            {
                "jobId": "",
                "tag": "POLICY_ANNUAL",
                "data": [
                    {"time": "时长", "role": "角色1", "word":"内容"},
                    {"time": "时长", "role": "角色2", "word":"内容"}
                    ]
            }
        ```
>- 当输入参数tag为`"POLICY_ANNUAL"`时，会通过`process_func(job_request)`调用函数`call_record_quality_process_func(job_request)`，执行任务。
>- 首先会从`get_chain(tag)`中获取`POLICY_ANNUAL`对应的提示词：`call_record_quality_prompt.txt"`，并且连线使用的ai模型，返回chain。
>- 用`chain.invoke({"content": conversation})`将处理过的对话数据传入到提示词的`{content}`位置，并用invoke获取ai的评价。
>- 最后进一步处理输出的评价，只提取\<judgement_basis>和\<result>。

### 通话违规检查
- 通话违规检查的tag为`"CALL_VIOLATION_REVIEW"`。

- `CALL_VIOLATION_REVIEW`是通过分析客户和经纪人之间的的沟通内容，检查聊天记录中是否出现敏感词和违规内容（在提示词中提供），并最终给出[违规,不违规]的结论。
    - data的主要内容有电话聊天数据。
>- 当输入参数tag为`"CALL_VIOLATION_REVIEW"`时，会通过`process_func(job_request)`调用函数`call_Violation_Review_process_func(job_request)`，执行任务。
>- 首先会从`get_chain(tag)`中获取`CALL_VIOLATION_REVIEW`对应的提示词：`call_violation_review_prompt.txt.txt"`，并且连线使用的ai模型，返回chain。
>- 用`chain.invoke({"content": conversation})`将处理过的对话数据传入到提示词的`{content}`位置，并用invoke获取ai的评价。
>- 最后进一步处理输出的评价，只提取\<thinking>和\<result>。