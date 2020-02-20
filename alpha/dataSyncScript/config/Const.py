import sys


class Const(object):
    class ConstError(TypeError):
        pass

    def __setattr__(self, key, value):
        # if self.__dict__.has_key(key):
        if key in self.__dict__:
            raise self.ConstError("Changing const.%s" % key)
        else:
            self.__dict__[key] = value

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.key
        else:
            return None

