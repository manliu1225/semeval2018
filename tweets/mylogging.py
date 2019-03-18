import logging

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
f = logging.Formatter(fmt='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt="%Y-%m-%dT%H:%M:%S")
# f = logging.Formatter(fmt='%(asctime)s %(name)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)', datefmt="%Y-%m-%dT%H:%M:%S")
for h in [logging.StreamHandler()]:
  h.setFormatter(f)
  h.setLevel(logging.DEBUG)
  root_logger.addHandler(h)
#end for

logger = logging.getLogger
