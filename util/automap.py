"""Automatically allocated dict

Like a default dict but with a set of predefined values
"""
# Works as follows:
# on every access, if key never accessed, allocate new value and return
# otherwise return that
# Useful for getting distinct colors when plotting


class AutoMap:

    def __init__(self, objects, wrap=True):
        self.mapping = {}
        self.idx = 0
        self.objects = list(objects)
        self.wrap = wrap

    def __getitem__(self, key):
        if key not in self.mapping:
            idx = self.idx
            if self.wrap:
                idx %= len(self.objects)
            self.mapping[key] = self.objects[idx]
            self.idx += 1
        return self.mapping[key]

    def __str__(self):
        return "AutoMap:" + repr(self.mapping)

    def __contains__(self, key):
        return key in self.mapping
        # self.__getitem__(key)
        # return True
