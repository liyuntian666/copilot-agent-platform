import uuid
from mongoengine import *
import threading
from cachetools import TTLCache

from entity import Parameter, Tool, Task
from utils import logger
import traceback


class TaskManager:
    """
    task实例和状态管理，提供task的生命周期管理方法，在数据库中新建任务、更新任务等。

    __init__ 方法获得访问数据库的连接和设置并发控制。
    """
    def __init__(self, mongo_host, mongo_db, mongo_port):
        self.mongoClient = connect(mongo_db, host=mongo_host, port=mongo_port)
        self.cache_lock = threading.Lock()

    def create_task(self):
        while True:
            # 创建任务 ID
            task_id = str(uuid.uuid4())
            task = Task.objects(task_id=task_id).first()
            if task is None:
                break
        task = Task()
        task.task_id = task_id
        task.status = 0
        task.edges = []
        task.nodes = []
        task.systemOutput = "初始化任务"
        task.save()
        return task_id

    def update_task(self, task_id, nodes, edges, systemOutput, isSuccess, isEnd):
        try:
            task = Task.objects.get(task_id=task_id)
            if not task:
                logger.warning(f"任务[{task_id}]不存在")
                return None

            logger.info(f"Updating task {task_id}: nodes={nodes}, edges={edges}, "
                        f"systemOutput={systemOutput}, isSuccess={isSuccess}, isEnd={isEnd}")
            if isEnd:
                task.isSuccess = isSuccess
                task.status = -1
            else:
                task.status = task.status + 1
                task.isSuccess = "任务实时执行调用情况"

            task.nodes = nodes
            task.edges = edges
            task.systemOutput = systemOutput

            task.save()

            return task_id
        except Exception as e:
            logger.error(f"任务[{task_id}]保存失败: {e}\n{traceback.format_exc()}")
        return None

    def get_task_by_id(self, task_id):
        try:
            task = Task.objects.get(task_id=task_id)
            if not task:
                logger.warning(f"任务[{task_id}]不存在")
                return None
            return task
        except Exception as e:
            logger.error(f"获取任务[{task_id}]失败: {e}\n{traceback.format_exc()}")
            return None


if __name__ == "__main__":
    taskManager = TaskManager('localhost', "tools", 27017)
    task_id = taskManager.create_task()
    logger.info(task_id)
