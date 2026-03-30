import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    # 获取当前节点名称，用于日志和任务状态记录
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行 LangGraph 节点: {current_node}")

    # 标记任务运行状态，用于任务监控/前端进度展示
    add_running_task(state.get("task_id",""), current_node) # 开始
    logger.info("--- BGE-M3 文本向量化处理启动 ---")

    try:
        # 1.获取要生存向量的chunks
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            logger.error(">>> chunks 数据无效，请检查数据格式")
            raise ValueError(">>> chunks数据无效，请检查数据格式")
        # 2. 给每个chunk生成向量
        # 2.1 获取嵌入式模型的客户端
        # 2.2 批量生成向量

        """
            1. 什么内容需要生成向量
                chunks - chunks - content - 生成向量
                用户 - 问题(华为手机怎么开机) - 向量 - 混合检索 - 向量  (item_name 华为手机 title 开机方式 content 三击屏幕侧方按钮，即可开机)
                      问题 -> item_name == || item_name 确保不会差错文档
                什么东西 content - 向量
                    item_name content(title + 内容)
                    f"商品名：item_name,介绍：content"
                    有中文，尽量使用中文标点符号
                    原则：核心词前置（前置集中力 权重 前128token)
            2. 列表能放一个字符串
                for - chunk - [content,content,content] -> 生成向量
                8192 - token -5（BGE-M3上下文窗口8192token，每个chunk约数百字，5条一批时保守估算，避免单次超限
        """
        final_chunks = [] # 存储处理完的chunk -> 带有向量
        batch_size = 5 # 需要运算 上下文窗口(token) / 块的大小（字符）

        for i in range(0, len(chunks),batch_size):
            # i+1 i+batch_size(步长)
            # 本批次处理的chunk
            batch_items = chunks[i:i+batch_size]
            # 定义当前批次的字符串
            current_texts = []
            for item in batch_items:
                item_name = item.get("item_name")
                item_content = item.get("content")
                # 原则：核心词前置
                item_text = f"商品：{item_name}，内容介绍：{item_content}"
                current_texts.append(item_text)
            # 当前批次生成的向量
            result = generate_embeddings(current_texts)
            # 当前批次的chunk添加向量即可
            # 完善chunk的属性添加稠密和稀疏向量
            for j, chunk in enumerate(batch_items):
                # 不改原始chunk，用copy()
                chunk_item = chunk.copy()
                # 保留原始数据不被污染，在副本上追加dense_vector（稠密）和sparse_vector（稀疏）两个向量字段
                chunk_item['dense_vector'] = result['dense'][j]
                chunk_item['sparse_vector'] = result['sparse'][j]
                final_chunks.append(chunk_item)
        state["chunks"] = final_chunks
        logger.info(f"--- BGE-M3 向量化处理完成，共处理{len(final_chunks)} 条文本切片 ---")
        add_done_task(state.get("task_id",""), current_node)    # 完成

    except Exception as e:
        # 捕获异常节点所有异常，记录错误堆栈，不中断整体流程
        logger.error(f"BGE-M3向量化节点执行失败：{str(e)}", exc_info=True)

    return state
