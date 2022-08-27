# a node class - each node intiate the right, left and the values.
class Node:
    def __init__(self, val):
        self.l = None
        self.r = None
        self.v = val

    def __str__(self):
        '''override to print the node as an dumb dict '''
        left  = '"null"' if self.l is None else str(self.l)
        right = '"null"' if self.r is None else str(self.r)
        val   = str(self.v)
        return f"""{'{'}"left": {left}, "right": {right}, "value": {val}{'}'}"""

# class of a binary search tree
class Tree:
    def __init__(self):
        self.root = None

    def __str__(self):
        return self.root.__str__()

    def getRoot(self):
        return self.root

    def add(self, val):
        if self.root is None:
            self.root = Node(val)
        else:
            self._add(val, self.root)

    def _add(self, val, node):
        if val < node.v:
            if node.l is not None:
                self._add(val, node.l)
            else:
                node.l = Node(val)
        else:
            if node.r is not None:
                self._add(val, node.r)
            else:
                node.r = Node(val)

    def find(self, val):
        if self.root is not None:
            return self._find(val, self.root)
        else:
            return None

    def _find(self, val, node):
        if val == node.v:
            return node
        elif (val < node.v and node.l is not None):
            return self._find(val, node.l)
        elif (val > node.v and node.r is not None):
            return self._find(val, node.r)

    def return_tree(self):
        import json
        return json.loads(self.__str__())
