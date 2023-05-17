from typing import IO

from inspect import isfunction, ismethod

from pathlib import Path

import logging
import time
import json


# class to throttle the number of times a function can be called synchronously
# within a given time period
class SynchronousThrottle:
	def __init__(self, call_count: int, period_in_minutes: int):
		self._call_count = call_count
		self._period_in_minutes = period_in_minutes
		self._wait_time = self.time_between_calls()
		self._last_call_time = None

		self._logging = logging.getLogger(__name__)
		self._logging.info(
			f'Created SynchronousThrottle with call count {self._call_count} per {self._period_in_minutes} '
			f'minutes with wait time {self._wait_time} seconds')

	def time_between_calls(self):
		return self._period_in_minutes * 60 / self._call_count

	def __call__(self, func, *args):
		now = time.time()
		if self._last_call_time is None or now - self._last_call_time > self._wait_time:
			self._last_call_time = now
			return func(*args)
		else:
			self._logging.info(f'Waiting {self._wait_time} seconds before calling {func}')
			time.sleep(self._wait_time)
			self._last_call_time = time.time()
			return func(*args)


# A class to cache the results of a function call in a file
class CachableCall:
	def __init__(self, cache_file_path: str):
		self._cache_file_path = cache_file_path
		self._cache = {}
		self._logging = logging.getLogger(__name__)
		self._cache_file = self._init_cache()
		self._closed = False

	def _init_cache(self) -> IO:
		cache_path = Path(self._cache_file_path)
		if cache_path.exists():
			with open(self._cache_file_path, 'r') as f:
				for line in f:
					data = json.loads(line)
					self._cache[data['key']] = data['value']
			self._logging.info(f'Loaded {len(self._cache)} entries from cache file {self._cache_file_path}')
		else:
			cache_path.touch()
			self._logging.info(f'Created new cache file {self._cache_file_path}')
		return open(self._cache_file_path, 'a', buffering=1)  # flush after every write

	def update_cache(self, key: str, value:str):
		self._cache_file.write(json.dumps({'key': key, 'value': value}) + '\n')
		self._cache_file.flush()
		self._logging.debug(f'Updated cache file {self._cache_file.name}')

	@staticmethod
	def fn_call_as_string(func, *args):
		arg_index = 0
		while not isfunction(func) and not ismethod(func):  # handle nested callable objects
			func = args[arg_index]
			arg_index += 1
		return f'{func.__name__}_{str(args[arg_index:])}'

	def __call__(self, func, *args):
		fn_key = CachableCall.fn_call_as_string(func, *args)
		if fn_key in self._cache:
			self._logging.debug(f'Cache hit for {fn_key}')
			return self._cache[fn_key]
		else:
			self._logging.debug(f'Cache miss for {fn_key}')
			if self._closed:
				raise Exception('Cache file has been closed')
			result = func(*args)
			self._cache[fn_key] = result
			self.update_cache(fn_key, result)
			return result

	def close(self):
		self._cache_file.close()
		self._closed = True
