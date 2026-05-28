from mongoengine import Document, StringField, ListField, DictField, IntField


class Task(Document):
    task_id = StringField()
    status = IntField()
    nodes = ListField(DictField())
    edges = ListField(DictField())
    isSuccess = StringField()
    systemOutput = StringField()

    def to_dict(self):
        """将 Task 对象转换为字典"""
        return {
            'task_id': self.task_id,
            'status': self.status,
            'nodes': self.nodes,
            'edges': self.edges,
            'isSuccess': self.isSuccess,
            'systemOutput': self.systemOutput
        }