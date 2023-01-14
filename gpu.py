import subprocess
import tensorflow as tf
import time


def __get_next_free_gpu(min_memory_mb):
	physical_devices_lst = tf.config.list_physical_devices('GPU')
	if not physical_devices_lst:
		raise Exception('No GPUs found')

	memory_lst = subprocess.run(
		['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
		check=True,
		stdout=subprocess.PIPE,
		universal_newlines=True
	).stdout.splitlines()
	gpu_index = -1
	for idx, mem in enumerate(memory_lst):
		if int(mem) >= min_memory_mb:
			gpu_index = idx
			break
	if gpu_index == -1:
		return None, memory_lst
	else:
		return physical_devices_lst[gpu_index], memory_lst


def set_gpu_to_next_available(min_memory_mb=22_000):
	gpu, memory_lst = __get_next_free_gpu(min_memory_mb=min_memory_mb)
	if gpu is None:
		raise Exception(
			f'Could not find a gpu with free memory of {min_memory_mb} MB. Current mem status (MB): {len(memory_lst)}')

	tf.config.set_visible_devices(gpu, 'GPU')
	return gpu


def wait_and_set_gpu_to_next_available(min_memory_mb=22_000, wait_in_sec=60, timeout_trails_int=None, log_b=True):
	gpu = None
	attempts = 0
	while gpu is None:
		gpu, memory_lst = __get_next_free_gpu(min_memory_mb=min_memory_mb)
		if gpu is None:
			if timeout_trails_int and attempts > timeout_trails_int:
				raise Exception(
					f'Failed to get a free GPU after {attempts} attempts.'
					f'Requested {min_memory_mb} MB. Current mem status (MB): {memory_lst}')
			time.sleep(wait_in_sec)
			attempts += 1
			if log_b:
				t = time.localtime()
				t_str = time.strftime("%H:%M:%S", t)
				print(
					f'{t_str}: Attempt {attempts} '
					f'Still waiting for an available gpu '
					f'with memory {min_memory_mb} MB. Current mem status (MB): {memory_lst}')
	if gpu:
		tf.config.set_visible_devices(gpu, 'GPU')
	return gpu
