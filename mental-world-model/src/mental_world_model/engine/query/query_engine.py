"""查询引擎（占位，尚未实现）。"""


class QueryEngine:
    """对心智关系图与索引执行查询。"""

    def query(self, expression):
        raise NotImplementedError("QueryEngine.query 尚未实现")

    def find_by_name(self, name):
        raise NotImplementedError("QueryEngine.find_by_name 尚未实现")

    def find_relations(self, item):
        raise NotImplementedError("QueryEngine.find_relations 尚未实现")
