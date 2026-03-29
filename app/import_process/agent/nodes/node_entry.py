import os
import sys
from pathlib import Path

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
    1. 进入节点的日志输出 【节点 + 核心参数】
        记录任务状态 【那个任务开始了】 -> 给前段推送信息（埋点）
    2. 参数校验（local_file_path -> 没有传入文件 -> end / local_dir -> 没有传如输出文件夹 -> 创建一个临时文件夹）
    3. 解析文件类型，修改state对应的参数 local_file_path -> md | pdf
        -> is_md_read_enabled True || is_pdf_read_enabled True
        -> md_path = local_file_path | pdf_path = local_file_path
        -> file_tile = 读取文件名
    4. 结束节点的日志输出 【节点 + 核心参数】
        记录任务状态 【那个任务结束了】 -> 给前段推送信息（埋点）
    """

    # 1. 进入节点的日志输出【节点 + 核心参数】
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] 开始执行了，现在状态为:{state}")
    add_running_task(state['task_id'], function_name)

    # 2. 进行必要的非空校验判定
    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.error(f"[{function_name}]检查发现没有输入文件，无法继续解析")
        return state

    # 3. 判定并且完成state属性赋值
    if local_file_path.endswith(".md"):
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    elif local_file_path.endswith(".pdf"):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    else:
        logger.error(f"[{function_name}]文件格式不是md或pdf，无法继续解析")

    # 提取file_title /xx/xxx/aa.pdf -> aa 为后期大模型没有识别出来当前文件对应item_name时用file_title兜底
    file_title_os = os.path.basename(local_file_path).split(".")[0]
    file_title = Path(local_file_path).stem # .stem方法只返回文件主体部分，去掉扩展名 .name返回完整文件名  .suffix 返回文件扩展名
    state['file_title'] = file_title

    # 4. 结束节点的日志输出【节点 + 核心参数】
    logger.info(f">>> [{function_name}] 结束了，现在状态为:{state}")
    add_done_task(state['task_id'], function_name)
    return state

    # 有日志证明这里面被调用就行，测试完整流程可以打开
    # 模拟简单的路由逻辑，防止报错 (仅 node_entry 需要)
    # if "local_file_path" in state:
    #     path = state["local_file_path"]
    #     if path.endswith(".pdf"):
    #         state["is_pdf_read_enabled"] = True
    #     elif path.endswith(".md"):
    #         state["is_md_read_enabled"] = True

# if __name__ == '__main__':
#
#     # 单元测试：覆盖不支持类型、MD、PDF三种场景
#     logger.info("===== 开始node_entry节点单元测试 =====")
#
#     # 测试1: 不支持的TXT文件
#     # test_state1 = create_default_state(
#     #     task_id="test_task_001",
#     #     local_file_path="联想海豚用户手册.txt"
#     # )
#     # node_entry(test_state1)
#
#     # # 测试2: MD文件
#     # test_state2 = create_default_state(
#     #     task_id="test_task_002",
#     #     local_file_path="小米用户手册.md"
#     # )
#     # node_entry(test_state2)
#     #
#     # # 测试3: PDF文件
#     test_state3 = create_default_state(
#         task_id="test_task_003",
#         local_file_path="万用表的使用.pdf"
#     )
#     node_entry(test_state3)
#
#     logger.info("===== 结束node_entry节点单元测试 =====")