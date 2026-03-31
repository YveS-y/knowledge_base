import sys

from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

def step_2_prepare_collections(state):
    """
    创建chunks对应的集合
    :param state:
    :return:
    """
    # 1. 获取milvus的客户端
    milvus_client = get_milvus_client()
    # 2. 判断是否存在集合（表），存在创建集合（表）
    if not milvus_client.has_collection(collenction_name=milvus_config.chunks_collection):
        # 创建集合
        # 3.1 创建集合对应的列信息
        schema = milvus_client.create_scheam(
            auto_id=True, # 主键自增
            enable_dynamic_field=True, # 动态字段
        )

        # 3.2 add field to schema
        # pk file_title item_name dense_vector sparse_vector
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 3.3 查询快，索引配置
        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector", # 给那个列创建索引 稠密
            index_name="dense_vector_index", # 索引的名字
            index_type="HNSW",  # 配置查找所用的算法
            metric_type="COSINE", # 配置县里那个匹配和对比的IP COSINE
            params={
                "M": 32,  # Maximum number of neighbors each node can connect to in the graph
                "efConstruction": 300 # or "DAAT_WAND" or "TAAT_NAIVE"
            },
        )
        """
           10000  M = 16  efConstruction = 200
           50000  M = 32  efConstruction = 300
           100000  M = 64  efConstruction = 400
           M:图中每个节点在层次结构的每个层级所能拥有的最大边数或连接数。M 越高，图的密度就越大，搜索结果的召回率和准确率也就越高，因为有更多的路径可以探索，但同时也会消耗更多内存，并由于连接数的增加而减慢插入时间。如上图所示，M = 5表示 HNSW 图中的每个节点最多与 5 个其他节点直接相连。这就形成了一个中等密度的图结构，节点有多条路径到达其他节点。
           efConstruction:索引构建过程中考虑的候选节点数量。efConstruction 越高，图的质量越好，但需要更多时间来构建。
        """
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_index",
            metric_type="IP",
            # 只计算可能得高分的向量，跳过大量的 0
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,
            index_params=index_params
        )
    return milvus_client


def step_3_delete_old_data(milvus_client, item_name):
    """
    删除旧数据，根据item_name删除
    :param milvus_client:
    :param item_name:
    :return:
    """
    milvus_client.delete(
        collection_name=CHUNKS_COLLECTION_NAME,
        filter=f"item_name=='{item_name}'"
    )
    milvus_client.load_collection(colllection_name=CHUNKS_COLLECTION_NAME)



def step_4_insert_collections(milvus_client, chunks):
    """
    插入集合的数据
    :param milvus_client:
    :param chunks: -> 主键回显
    :return:
    """
    insert_result = milvus_client.insert(collection_name=CHUNKS_COLLECTION_NAME,data=chunks)
    # 成功插入了几条
    insert_count = insert_result.get("insert_count",0)
    logger.info(f"完成了数据插入，成功插入了 {insert_count} 条数据")

    # 获取回显的ids
    ids = insert_result.get("ids",[])

    if ids and len(ids) == len(chunks):
        for index, chunk in enumerate(chunks):
            chunk['chunk_id'] = ids[index]
    return chunks


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    # 1. 进入的日志和任务状态的配置
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为: {state}")
    add_running_task(state['task_id'], function_name)
    try:
        # 1.获取输入的数量（校验）
        chunks = state.get('chunks')
        if not chunks:
            logger.error(f">>> [{function_name}]没有chunks数据，请检查！")
            raise ValueError("没有chunks数据")
        # 2. 没有集合，要创建集合 collection (filed,index,collection)
        milvus_client = step_2_prepare_collections(state)
        # 3. 删除旧数据
        step_3_delete_old_data(milvus_client, chunks[0]['item_name'])
        # 4.插入chunks数据
        with_id_chunks = step_4_insert_collections(milvus_client,chunks)

        state['chunks'] = with_id_chunks
    except Exception as e:
        logger.error(f">>> [{function_name}]导入chunks对应的向量数据库发生了异常，异常信息：{e}")
        raise
    finally:
        logger.info(f">>> [{function_name}]结束了！现在的状态为：{state}")
        add_done_task(state['task_id'], function_name)

    return state