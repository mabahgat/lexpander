import json

from utils import SynchronousThrottle, CachableCall
import time


# test SynchronousThrottle class
def test_synchronous_throttle_return():
	def test_func(s):
		return f'test_{s}'

	throttle = SynchronousThrottle(1, 1)
	assert throttle(test_func, 'string') == test_func('string')


def test_synchronous_throttle_wait():
	def test_func():
		return f'test'

	throttle = SynchronousThrottle(20, 1)
	throttle(test_func)
	call_time = time.time()
	throttle(test_func)
	assert time.time() - call_time >= 3


def test_cachable_call_return(tmp_path):
	def test(param: str):
		return f'test_{param}'
	tmp_file = tmp_path / 'cache.jsonl'

	cachable_call = CachableCall(str(tmp_file))
	assert cachable_call(test, 'string') == test('string')


def test_cachable_call_caching(tmp_path):
	def test(param: str):
		return f'test_{param}'
	tmp_file = tmp_path / 'cache.jsonl'

	cachable_call = CachableCall(str(tmp_file))
	assert tmp_file.exists()

	cachable_call(test, 'string')
	assert len(cachable_call._cache) == 1
	print(cachable_call._cache)
	assert cachable_call._cache['test_(\'string\',)'] == 'test_string'

	# read jsonl file and verify it has one jsonl entry
	with open(tmp_file, 'r') as f:
		entries = [json.loads(line) for line in f.readlines()]
		assert len(entries) == 1
		assert entries[0]['key'] == 'test_(\'string\',)'


def test_cachable_call_with_throttling(tmp_path):
	def test(param: str):
		return f'test_{param}'
	tmp_file = tmp_path / 'cache.jsonl'

	cachable_call = CachableCall(str(tmp_file))
	throttle = SynchronousThrottle(20, 1)

	value = cachable_call(throttle, test, 'string')
	assert value == test('string')
