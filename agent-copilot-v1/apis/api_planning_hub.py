import json

from apis.api_selection_hub import ApiSelectionHub
from models import LargeLanguageModel
from param_extraction.param_extraction_hub import ParamExtractionHub
from tasks import TaskManager,GenerateTaskHub
from tools import ToolSummaryHub, ToolUseHub
from utils import logger, TASK_ERROR_CODE, TASK_SUCCESS_CODE, RESPONSE_STATUS_CODE_SUCCESS, api_result_max_length, \
    api_result_max_threshold
import traceback

class ApiPlanningHub:
    def __init__(self, milvus_uri, model_path, milvus_db_name, model, temperature, top_p,
                mongo_host, mongo_db, mongo_port, topK, api_url, api_key):
        """
        初始化 ApiPlanningHub 类的实例。

        :param milvus_uri: 连接地址，通常用于指定服务的访问地址
        :param model_path: 模型文件的路径
        :param milvus_db_name: 数据库的名称
        :param model: 使用的LLM名称
        :param temperature: 采样温度，用于控制生成文本的随机性
        :param top_p: 核采样概率，用于控制生成文本的多样性
        :param mongo_host: 主机地址，通常用于数据库或服务的连接
        :param mongo_db: 数据库标识，可能用于指定具体的数据库
        :param mongo_port: 端口号，用于网络连接
        :param topK: 检索时返回的前 K 个结果
        """
        self.apiSelectionHub = ApiSelectionHub(milvus_uri, model_path, milvus_db_name, model, temperature, top_p,
                                                mongo_host, mongo_db, mongo_port, api_url, api_key)
        self.topK = topK
        self.paramExtractionHub = ParamExtractionHub(model, temperature, top_p, api_url, api_key)
        self.toolSummaryHub = ToolSummaryHub(model, temperature, top_p, api_url, api_key)
        self.toolUseHub = ToolUseHub("")
        self.generateTaskHub = GenerateTaskHub(model, temperature, top_p, api_url, api_key, mongo_host, mongo_db, mongo_port, milvus_uri, milvus_db_name)
        self.taskManager = TaskManager(mongo_host, mongo_db, mongo_port)
        self.llm = LargeLanguageModel(api_url, api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def generate_output(self, text):
        prompt = " 请将以下句子进行润色成通顺的话，请直接输出结果：\n\n\n" + text
        print(prompt)
        result = self.llm.chat_completions(prompt, self.model, self.temperature, self.top_p)
        return result

    def convert_inter_result(self, task_id, results, model_output, isEnd):
        nodes = []
        sub_nodes = []
        edges = []
        index = 1
        sub_index = 10000
        isSuccess = "正常调用链"
        if model_output == "当前API无法实现用户需求":
            self.taskManager.update_task(task_id, [], [], model_output, "异常调用链", isEnd)
            return
        for tmp in results:
            if tmp["code"] != TASK_SUCCESS_CODE:
                isSuccess = "异常调用链"
            nodes.append({
                "id": str(index),
                "name": str(index) + "_" + tmp["tool"],
                "label": tmp["tool"],
                "group": tmp["tool"],
                "task_description": tmp["query"],
                "params": json.dumps(tmp["param"], ensure_ascii=False),
                "result": json.dumps(tmp["result"], ensure_ascii=False),
            })
            for tmptmp in tmp["missing_param"]:
                sub_nodes.append({
                    "id": str(sub_index),
                    "name": str(sub_index) + "_" + tmptmp["tool"],
                    "label": tmptmp["tool"],
                    "group": tmptmp["tool"],
                    "task_description": tmptmp["task_description"],
                    "params": json.dumps(tmp["param"], ensure_ascii=False),
                    "result": json.dumps(tmp["result"], ensure_ascii=False),
                })
                edges.append(
                    {
                        "source": str(sub_index), "target": str(index), "value": '缺省参数逆向API规划', "symbolSize": [5, 20],
                        "label": {"show": False}
                    }
                )
                sub_index += 1
            index += 1
        if len(nodes) == 1:
            nodes = nodes + sub_nodes
            self.taskManager.update_task(task_id, nodes, edges, model_output, "正常调用链路", isEnd)
            return
        for i in range(len(nodes) - 1):
            edges.append(
                {
                    "source": nodes[i]["id"], "target": nodes[i + 1]["id"], "value": '正向API规划', "symbolSize": [5, 20],
                    "label": {"show": False}
                }
            )
        nodes = nodes + sub_nodes
        self.taskManager.update_task(task_id, nodes, edges, model_output, isSuccess, isEnd)
        return

    def loop_validate(self, api_chains):
        """
        检测API调用链中是否存在循环调用。主要逻辑如下：
        遍历API调用链，跟踪当前API名称和连续相同结果的计数
        当遇到不同API时重置计数器
        当连续3次出现相同API且结果相同时，判定为循环调用并返回False
        如果没有检测到循环则返回True
        该函数通过比较相邻API调用的结果来识别可能的无限循环情况。
        """
        current_api_name = ""
        number = 0
        for i in range(len(api_chains)):
            api_response = api_chains[i]
            if api_response["tool"] != current_api_name:
                current_api_name = api_response["tool"]
                number = 1
            else:
                if api_chains[i - 1]["result"] == api_response["result"]:
                    number += 1
                if number == 3:
                    return False
        return True

    def supplement_parameters(self, new_query, missing_param):
        """
        补充缺失的参数值。
        该方法根据新的查询语句和缺失的参数名，尝试从工具调用的结果中获取缺失的参数值。
        首先通过 `apiSelectionHub` 获取合适的工具，然后使用 `paramExtractionHub` 提取参数，
        调用工具并解析返回结果，若结果中包含缺失的参数，则返回该参数值和工具名称。
        输入：
            :param new_query: 新的查询语句，用于获取缺失参数的值
            :param missing_param: 缺失的参数名
        :return:
            若找到缺失参数的值，返回参数值和工具名称、汇总响应结果；否则返回3个 None
        """
        tool = self.apiSelectionHub.get_tool_coarse_and_fine(new_query, None, topK=self.topK)
        if tool is None:
            return None, None, None
        params, new_missing_param = self.paramExtractionHub.extraction_params(new_query, tool)
        if len(new_missing_param) != 0:
            return None, None, None

        single_tool_response = self.toolUseHub.tool_use(tool, params)

        if single_tool_response.status_code != RESPONSE_STATUS_CODE_SUCCESS:
            return None, None, None

        if len(single_tool_response.text) != 0:
            results = json.loads(single_tool_response.text)

            if isinstance(results, list) and len(results) != 0:
                results = results[0]
            if missing_param in results:
                return results[missing_param], tool.name_for_human, {
                    "code": TASK_SUCCESS_CODE,
                    "tool": tool.name_for_human,
                    "result": single_tool_response.text,
                    "param": params,
                    "query": new_query,
                    "task_description": new_query
                }
            else:
                return None, None, None
        else:
            return None, None, None



    def single_api_planning(self, query, raw_query="", api_chains=None):
        """
        单API规划函数，用于处理单个查询的API调用流程。

        此函数接收一个查询语句，根据查询选择合适的工具，提取参数，处理缺失参数，
        并最终调用工具获取结果。如果在处理过程中出现问题或找不到合适的工具，
        则返回相应的错误信息。

        代码流程：
        1. 通过 `apiSelectionHub` 根据查询语句获取合适的工具。
        2. 若未找到工具，返回包含错误信息的结果。
        3. 若找到工具，使用 `paramExtractionHub` 提取参数和缺失参数。
        4. 若没有缺失参数，直接调用工具并返回结果。
        5. 若有缺失参数，使用 `generateTaskHub` 生成新查询，调用 `supplement_Parameters` 补充缺失参数。
        6. 若所有缺失参数都补充成功，调用工具并返回结果；若有缺失参数补充失败，返回错误信息。

        :param
            query: 输入的查询语句，用于描述用户的需求
            raw_query：用户的原始查询
        :return:
            一个字典，包含处理结果的相关信息，如状态码、工具名称、结果内容、缺失参数信息、参数列表和任务描述等
        """
        logger.debug(f"[{query}]开始单API检索....")
        tool = self.apiSelectionHub.get_tool_coarse_and_fine(query, None, topK=self.topK)

        if tool is None:
            return {
                "code": TASK_ERROR_CODE,
                "result": "",
                "tool": "异常调用节点-无法寻找到合适的API工具",
                "missing_param": [],
                "param": {},
                "query": query,
                "task_description": "无法寻找到合适的API工具"
            }
        else:
            params, missing_params = self.paramExtractionHub.extraction_params(query + " " + raw_query, tool,api_chains)
            logger.debug(f"参数提取[{query + ' ' + raw_query}]结果：params:{params}，missing_params：{missing_params}")
            new_params = params.copy()
            if len(missing_params) == 0:
                logger.debug(f"[{query}]未缺失参数，直接工具调用")
                inject_flag, reason = self.generateTaskHub.gen_judge_task(query, tool, new_params)
                if inject_flag:
                    return {
                        "code": TASK_ERROR_CODE,
                        "result": "inject",
                        "tool": "异常调用节点-提示注入攻击",
                        "missing_param": [],
                        "param": {},
                        "query": query,
                        "task_description": reason
                    }
                single_tool_response = self.toolUseHub.tool_use(tool, new_params)
                if single_tool_response.status_code != RESPONSE_STATUS_CODE_SUCCESS:
                    return {
                        "code": TASK_ERROR_CODE,
                        "result": "",
                        "tool": "异常调用节点-调用外部系统失败",
                        "missing_param": [],
                        "param": {},
                        "query": query,
                        "task_description": "调用外部系统失败"
                    }
                return {
                    "code": TASK_SUCCESS_CODE,
                    "tool": tool.name_for_human,
                    "result": single_tool_response.text,
                    "missing_param": [],
                    "param": params,
                    "query": query,
                    "task_description": query
                }
            else:
                logger.debug(f"[{query}]缺失参数，进行参数的补齐....")
                missing_params_supplemented = []
                for missing_param in missing_params:
                    gen_param_query = self.generateTaskHub.gen_param_task(query,
                                                                    json.dumps(params, ensure_ascii=False, indent=4),
                                                                    f'{missing_param.name}: {missing_param.description}')
                    logger.debug(f"[{query}]参数生成或补齐任务的描述词为：{gen_param_query}")
                    supplement_param, supplement_param_tool, supplement_param_result = self.supplement_parameters(
                        gen_param_query, missing_param.name)
                    if supplement_param is not None:
                        logger.debug(f"[{query}]的参数{supplement_param}已补齐：\n{supplement_param_tool}\n{supplement_param_result}")
                        params[missing_param.name] = supplement_param
                        missing_params_supplemented.append(supplement_param_result)
                    else:
                        return {
                            "code": TASK_ERROR_CODE,
                            "result": "missing_param",
                            "tool": "异常调用节点-缺少必要参数",
                            "missing_param": [],
                            "param": {},
                            "query": query,
                            "task_description": f"通过参数补全Query:{gen_param_query} 无法为 {tool.name_for_human} API 补全缺少参数 {missing_param.name}"
                        }

                inject_flag, reason = self.generateTaskHub.gen_judge_task(query, tool, params)
                if inject_flag:
                    return {
                        "code": TASK_ERROR_CODE,
                        "result": "inject",
                        "tool": "异常调用节点-提示注入攻击",
                        "missing_param": [],
                        "param": {},
                        "query": query,
                        "task_description": reason
                    }

                logger.debug(f"[{query}]准备进行工具{tool.operationId}调用,参数为：{params}")
                single_tool_response = self.toolUseHub.tool_use(tool, params)
                logger.debug(f"[{query}]工具{tool.operationId}调用完成,响应为：{single_tool_response}")
                if single_tool_response.status_code != RESPONSE_STATUS_CODE_SUCCESS:
                    return {
                        "code": TASK_ERROR_CODE,
                        "result": "",
                        "tool": "异常调用节点-调用外部系统失败",
                        "missing_param": [],
                        "param": {},
                        "query": query,
                        "task_description": "调用外部系统失败"
                    }
                else:
                    return {
                        "code": TASK_SUCCESS_CODE,
                        "tool": tool.name_for_human,
                        "result": single_tool_response.text,
                        "missing_param": missing_params_supplemented,
                        "param": params,
                        "query": query,
                        "task_description": query
                    }

    def apis_planning(self, query, task_id):
        """
        多API规划函数，用于处理单个或多个查询的API调用流程。
        此函数接收一个查询语句，根据查询判断是否为单任务。若为单任务，直接调用单API规划函数；
        若为多任务，则循环调用单API规划函数，直到任务处理完成。最终返回API调用结果链。

        代码流程：
        1. 调用 `generateTaskHub.gen_root_task` 判断是否为单任务并获取根任务描述。
        2. 若为单任务，调用 `single_api_planning` 处理并将结果添加到结果链中。
        3. 若为多任务，进入循环，不断调用 `single_api_planning` 处理任务，
        并通过 `generateTaskHub.gen_task_from_context` 判断是否还有后续任务。
        4. 若最后还有剩余任务描述，再次调用 `single_api_planning` 处理并添加到结果链中。
        5. 返回完整的API调用结果链。

        :param
            query:输入的查询语句，用于用户描述的需求
            task_id:当前任务内部编号
        :return:
            一个列表，包含一个或多个字典，每个字典表示一次API调用的处理结果，
            包含状态码、工具名称、结果内容、缺失参数信息、参数列表和任务描述等
        """

        def process_single_api_invoke(curr_task_desc):
            api_invoke_result = self.single_api_planning(curr_task_desc, raw_query,api_chain)
            logger.debug(f"[{query}]工具调用结果：{api_invoke_result}")
            api_chain.append(api_invoke_result)
            if api_invoke_result["code"] == TASK_ERROR_CODE:
                logger.debug(f"[{query}]工具调用失败，中止任务...")
                self.convert_inter_result(task_id, api_chain, api_invoke_result["task_description"], True)
            # 解决调用函数时，返回结果过多的问题
            api_result_length_limit = int(api_result_max_length * api_result_max_threshold)
            api_invoke_result_length = len(api_invoke_result["result"])
            if api_invoke_result_length >= api_result_length_limit:
                logger.warning(f"[{query}]API调用结果长度超出限制，截取长度{api_result_length_limit}...")
                api_result_fact = api_invoke_result["result"][:api_result_length_limit]
                api_invoke_result["result"] = f"API调用结果长度为{api_invoke_result_length}超出限制，进行了截取，保留的内容为：{api_result_fact}"
            logger.debug(f"[{query}]工具实际调用结果：{api_invoke_result}")
            return api_invoke_result

        def process_mul_task_api_invoke(curr_task_desc):
            api_invoke_result = process_single_api_invoke(curr_task_desc)
            logger.debug(f"[{query}]工具调用结果：{api_invoke_result}")
            if not self.loop_validate(api_chain):
                logger.debug(f"[{query}]存在工具循环调用，中止任务...")
                self.convert_inter_result(task_id, api_chain, "循环调用错误", True)
                return False,api_invoke_result
            return True, api_invoke_result

        raw_query = query
        api_chain = []
        is_single_task, root_task_description = self.generateTaskHub.gen_root_task(query)
        logger.info(f"[{query}]任务是否单任务:{is_single_task},当前任务描述为：{root_task_description}")
        if is_single_task:
            single_task_result = process_single_api_invoke(query)
            if single_task_result["code"] == TASK_ERROR_CODE:
                return api_chain, single_task_result["task_description"]
            summary = self.toolSummaryHub.tool_summary(query, api_chain)
            logger.debug(f"[{query}]任务完成，总结任务概要为：{summary}")
            self.convert_inter_result(task_id, api_chain, summary, True)
            return api_chain, summary
        else:
            is_complete = False
            task_description = root_task_description
            #任务未完成且任务描述不为空，则循环处理任务
            while not is_complete and task_description is not None:
                loop_validate_result,single_task_result = process_mul_task_api_invoke(task_description)
                if single_task_result["code"] == TASK_ERROR_CODE:
                    return api_chain, single_task_result["task_description"]
                if not loop_validate_result:
                    return api_chain, single_task_result["task_description"]
                logger.debug(f"[{query}]任务的子任务完成：{task_description}")
                self.convert_inter_result(task_id, api_chain, f"已处理完成子任务 {task_description}", False)
                is_complete, task_description = self.generateTaskHub.gen_from_context_task(query, api_chain)
                logger.debug(f"[{query}]任务是否完成：{is_complete}，后续任务描述为：{task_description}")

            # 任务标注已完成但任务描述不为空，处理剩余任务
            if task_description is not None:
                loop_validate_result,single_task_result = process_mul_task_api_invoke()
                if single_task_result["code"] == TASK_ERROR_CODE:
                    return api_chain, single_task_result["task_description"]
                if not loop_validate_result:
                    return api_chain, "循环调用错误"
        summary = self.toolSummaryHub.tool_summary(query, api_chain)
        logger.debug(f"[{query}]任务完成，总结任务概要为：{summary}")
        self.convert_inter_result(task_id, api_chain, summary, True)
        return api_chain, summary


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    import uuid
    load_dotenv()
    milvus_uri = os.getenv("milvus_uri", "http://localhost:19530")
    model_path = os.getenv("model_path", "model")
    milvus_db_name = os.getenv("milvus_db_name", "tool_db")
    model = os.getenv("model", "qwen-max-0919")
    temperature = float(os.getenv("temperature", "0.01"))
    top_p = float(os.getenv("top_p", "0.01"))
    mongo_host = os.getenv("mongo_host", "127.0.0.1")
    mongo_db = os.getenv("mongo_db", "tools")
    mongo_port = int(os.getenv("mongo_port", "27017"))
    topK = int(os.getenv("topK", "5"))
    api_key = os.getenv("api_key", "")
    base_url = os.getenv("base_url", "")
    task_id = str(uuid.uuid4())
    apiPlanningHub = ApiPlanningHub(milvus_uri, model_path, milvus_db_name, model, temperature, top_p, 
                                    mongo_host, mongo_db, mongo_port, topK, base_url,api_key)


    # # 单工具测试
    # # 测试一
    # response = apiPlanningHub.single_api_planning("查询产品名为苹果的产品信息")
    # logger.info(response)

    ## 测试二
    # response = apiPlanningHub.single_api_planning("我想知道苹果的产品有关信息")
    # logger.info(response)

    # # 测试三
    # response = apiPlanningHub.single_api_planning("查询能配送到北京的物流供应商")
    # logger.info(response)

    # # 测试四
    # response = apiPlanningHub.single_api_planning("查询ID为2的物流供应商信息")
    # logger.info(response)

    # # 测试五
    # response = apiPlanningHub.single_api_planning("查询ID为2的物流供应商信息")
    # logger.info(response)

    # badCASE 类型目前只能适配字符串
    # 有些API检索有混淆 查询叫做京东的物流供应商信息 这个跟 getOrdersBySupplierId 混淆
    # 时间类型适配

    # 测试六
    # response = apiPlanningHub.single_api_planning("查询ID为2的订单信息")
    # logger.info(response)

    # # 测试七
    # response = apiPlanningHub.single_api_planning("查询ID为3的产品信息")
    # logger.info(response)

    # # 测试八
    # response = apiPlanningHub.single_api_planning("查询配送区域为北京的物流供应商信息")
    # logger.info(response)

    # # 测试用多工具函数调用单工具
    # # 测试一
    # import uuid
    # task_id = str(uuid.uuid4())
    # response = apiPlanningHub.apis_planning("查询产品名为苹果的产品信息",task_id)
    # logger.info(response)

    # # 多工具测试
    # # 调用两个查询API，这两个API之间没有依赖关系
    # # 样例一  - success but data error
    # response = apiPlanningHub.apis_planning("请先查询苹果的产品信息，再查询与能够配送北京的物流供应商信息",task_id)
    # logger.info(response)

    # # 样例二？
    # response = apiPlanningHub.apis_planning("请先查询苹果的产品信息，再查询与能够配送北京的物流供应商信息")
    # 调用一个查询API，查询API的返回值是update或者create query的输入值

    # 多工具测试3
    # 调用一个查询API，然后发现该API的请求参数需要依赖其他API

    # 样例三 - success but data error
    # response, summary = apiPlanningHub.apis_planning("请查询苹果产品库存信息",task_id)
    # logger.info(response)
    # logger.info(summary)

    # # 样例四 -success but data error
    # response = apiPlanningHub.single_api_planning("请创建一个订单，该订单的产品名称为苹果，数量为10，配送区域为北京，物流供应商ID为3的订单")
    # logger.info(response)

    # 多工具测试4
    # 异常处理

    # 多工具测试4
    # 调用一个查询API，然后发现该API的请求参数需要依赖其他API，然后再做另外一个API的调用

    # 每组测试两个query
