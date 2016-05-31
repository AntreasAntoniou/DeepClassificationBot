# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import functools
from multiprocessing import TimeoutError
import multiprocessing.pool
import time


# courtesy of http://stackoverflow.com/a/35139284/20226
def timeout(max_timeout):
    """Timeout decorator, parameter in seconds."""
    def timeout_decorator(f):
        """Wrap the original function."""
        @functools.wraps(f)
        def func_wrapper(self, *args, **kwargs):
            """Closure for function."""
            pool = multiprocessing.pool.ThreadPool(processes=1)
            async_result = pool.apply_async(f, (self,) + args, kwargs)
            timeout = kwargs.pop('timeout_max_timeout', max_timeout) or max_timeout
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(timeout)
        return func_wrapper
    return timeout_decorator
