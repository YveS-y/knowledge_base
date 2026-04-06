from typing import Dict, List       # Python 内置类型提示工具
from .sse_utils import push_to_session  # 从同目录下的 see_utils 导入推送函数（SSE 推流用）

"""
整体数据流
节点开始执行
    -> add_running_task(task_id, "node_pdf_to_md", is_stream=True)
    -> 前端收到：running=["PDF转Markdown"], done=["检查文件"]
节点执行完成
    -> add_done_task(task_id, "node_pdf_to_md", is_stream=True)
    -> 前端收到: running=[], done=["检查文件", "PDF转Markdown"]
全部完成
    -> update_task_status(task_id, "completed", push_queue=True)
    -> clear_task(task_id)
"""

# ---------------------------
# 内存态任务追踪（单进程）
# 四个内存字典（全局“数据库”）
# ---------------------------
# key: task_id
# value: 节点名列表（原始英文/节点ID）
_tasks_running_list: Dict[str, List[str]] = {}
# 存“正在运行”的节点，结构：{ "task_001":["node_entry", "node_pdf_to_md"] }
_tasks_done_list: Dict[str, List[str]] = {}
# 存"已完成"的节点，结构："task_001":["node_entry"] }

# key: task_id
# value: status 字符串（如 pending/processing/completed/failed）
_tasks_status: Dict[str, str] = {}
# 存任务整体状态，结构：{ "task_001": "processing" }

# key: task_id
# value: 任务结果（例如 query 的 answer）
_tasks_result: Dict[str, Dict[str, str]] = {}
# 存任务结构，结构：{ "task_001": {"answer": "xxx", "error": ""} }

TASK_STATUS_PENDING = "pending"         # 待处理
TASK_STATUS_PROCESSING = "processing"   # 处理中
TASK_STATUS_COMPLETED = "completed"     # 已完成
TASK_STATUS_FAILED = "failed"           # 失败

# 节点名 -> 中文名映射（用于前端展示）
# 说明：这里的 key 应与 LangGraph 的 add_node("xxx", ...) 中的节点名一致。
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_kg": "导入知识图谱",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    # --- Query 流程节点（kb/query_process/main_graph.py）---
    "node_item_name_confirm": "确认问题产品",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_query_kg": "查询知识图谱",
    "node_join": "多路搜索合并",
}
# 工具函数部分
# 初始化保障
def _ensure_task(task_id: str) -> None:
    """确保 task_id 对应的数据结构已初始化。"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []   # 不存在就建空列表
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}

# 翻译节点名
def _to_cn(node_name: str) -> str:
    """将节点名转换为中文展示名；若无映射则返回原名（不报错）。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)

# 标记节点“开始执行”
def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    添加“正在运行”的节点任务。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """
    # 先确保初始化
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    # 避免重复追加
    if node_name not in running:
        running.append(node_name)
    # 如果需要实时推流
    if is_stream:
        # 立刻把最新状态推给前端
        task_push_queue(task_id)

# 标记节点“执行完成
def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """
    添加“已完成”的节点任务。

    注意：添加已完成任务时，会把同名的“正在运行”任务删除。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """
    _ensure_task(task_id)

    # 1) 从 running 中移除同名节点（可能出现重复，移除所有）
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]

    # 2) 追加到 done（保持完成顺序），避免重复
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)
    # 推流通知前端
    if is_stream:
        task_push_queue(task_id)

# 存结果
def set_task_result(task_id: str, key: str, value: str) -> None:
    """
    存储任务结果字段（如 answer / error）。
    """
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value

# 取结果
def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """
    获取任务结果字段（如 answer / error）。
    """
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)

# 获取当前任务状态
def get_task_status(task_id: str) -> str:
    """
    获取当前任务状态。

    参数：
    - task_id: 任务ID

    返回：
    - str: 状态名称；如果未设置过则返回空字符串
    """
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """
    获取已完成节点列表（中文展示）。


    """
    _ensure_task(task_id)
    done = _tasks_done_list.get(task_id, [])
    return [_to_cn(n) for n in done]


def get_running_task_list(task_id: str) -> List[str]:
    """
    获取正在运行节点列表（中文展示）。

    """
    _ensure_task(task_id)
    running = _tasks_running_list.get(task_id, [])
    return [_to_cn(n) for n in running]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """
    更新任务状态。

    参数：
    - task_id: 任务ID
    - status_name: 状态名称（字符串）
    """
    _tasks_status[task_id] = status_name
    if push_queue:
        task_push_queue(task_id)

# 核心推流函数
# 把当前快照打包成一条 SSE 消息推给前端，前端收到后就能实时更新进度条
def task_push_queue(task_id: str):
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


# 清理内存
# 任务彻底结束后清理内存，防止内存泄漏（纯内存存储，进程重启就丢失）
def clear_task(task_id: str):
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)