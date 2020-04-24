import logging
import os


class Logger(object):
    def __init__(self, module_name):
        self.module = module_name
        if os.path.isdir("/var/www/html/"):
            self.log_file = "/tmp/brace_log.log"
        else:
            self.log_file = "./brace_log.log"

        self.trace_format = '%(asctime)s %(name)s: %(levelname)s %(filename)s[line:%(lineno)d]  %(message)s'

    def init_logger(self):
        self.logger = logging.getLogger(self.module)
        self.logger.setLevel('DEBUG')
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        ch = logging.StreamHandler()

        formatter = logging.Formatter(
            fmt=self.trace_format,
            datefmt="%Y/%m/%d %X",
        )

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        return self.logger


log = Logger("alpha")
alpha_logger = log.init_logger()


if __name__ == '__main__':
    alpha_logger.info("Hello")
    alpha_logger.error("Hello")
