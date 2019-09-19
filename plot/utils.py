
# Works as follows:
# on every access, if key never accessed, allocate new value and return
# otherwise return that


class AutoMap:

    def __init__(self, objects):
        self.mapping = {}
        self.idx = 0
        self.objects = list(objects)

    def __getitem__(self, key):
        if key not in self.mapping:
            self.mapping[key] = self.objects[self.idx]
            self.idx += 1
        return self.mapping[key]

    def __repr__(self):
        return repr(self.mapping)

    def __contains__(self, key):
        self.__getitem__(key)
        return True
