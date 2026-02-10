class Wrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, args, shared_args_idx):
        while shared_args_idx.value < len(args):
            self.func(*args[shared_args_idx.value])
            shared_args_idx.value += 1


def split_list(lst, num_chunks):
    n = len(lst)
    base_size, remainder = divmod(n, num_chunks)
    sizes = [base_size + (1 if i < remainder else 0) for i in range(num_chunks)]

    chunks = []
    start = 0
    for size in sizes:
        chunks.append(lst[start: start + size])
        start += size
    return chunks


class ProcessPool:
    def __init__(self, task_func, task_args: list[tuple], n_processes: int = 16, timeout: float = 5):
        self.n_processes = n_processes
        self.timeout = timeout
        self.task_func = task_func
        self.task_args = task_args

    def run(self):
        import time
        from multiprocessing import Process, Value
        import multiprocessing
        from tqdm import tqdm

        default_method = multiprocessing.get_start_method()
        multiprocessing.set_start_method("fork", force=True)

        pbar = tqdm(total=len(self.task_args), desc="Generating meshes")
        task_args = split_list(self.task_args, self.n_processes)
        shared_args_indicies = [Value('i', 0) for _ in range(self.n_processes)]
        last_args_indicies = [0 for _ in range(self.n_processes)]
        pool = [Process(target=Wrapper(self.task_func), args=(task_args[i], shared_args_indicies[i]), daemon=True)
                for i in range(self.n_processes)]
        unprocessed_args = []

        for process in pool:
            process.start()

        n_processes_running = self.n_processes

        while n_processes_running:
            time.sleep(self.timeout)
            for i in range(self.n_processes):
                if pool[i] is None:
                    continue

                if shared_args_indicies[i].value >= len(task_args[i]):
                    process = pool[i]
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
                    pool[i] = None
                    n_processes_running -= 1
                elif shared_args_indicies[i].value == last_args_indicies[i]:
                    hang_process = pool[i]
                    hang_process.terminate()
                    hang_process.join(timeout=5)
                    if hang_process.is_alive():
                        hang_process.kill()
                        hang_process.join()

                    unprocessed_args.append(task_args[i][shared_args_indicies[i].value])
                    # with shared_args_indicies[i].get_lock():
                    shared_args_indicies[i].value += 1
                    new_process = Process(
                        target=Wrapper(self.task_func), args=(task_args[i], shared_args_indicies[i]), daemon=True
                    )
                    new_process.start()
                    pool[i] = new_process

                last_args_indicies[i] = shared_args_indicies[i].value

            pbar.update(sum(last_args_indicies) - pbar.n)

        pbar.close()

        multiprocessing.set_start_method(default_method, force=True)

        return unprocessed_args


def py_str_to_mesh_file(py_string, save_path):
    if py_string is None:
        return
    try:
        namespace = {}
        exec(py_string, namespace)
        compound = namespace['r'].val()
        assert len(compound.Faces()) > 2
        compound.export(save_path, tolerance=0.001, angularTolerance=0.1)
    except:
        pass


def generate_meshes(py_strings_and_save_paths: list[tuple[str, str]], timeout=5):
    import os

    task_args = [None] * len(py_strings_and_save_paths)

    for i in range(len(py_strings_and_save_paths)):
        path = py_strings_and_save_paths[i][0]
        if not os.path.exists(path):
            task_args[i] = (None, py_strings_and_save_paths[i][1])
        else:
            with open(path, 'r', encoding='utf-8') as f:
                task_args[i] = (f.read(), py_strings_and_save_paths[i][1])

    pool = ProcessPool(
        task_func=py_str_to_mesh_file,
        task_args=task_args,
        n_processes=16,
        timeout=timeout,
    )
    pool.run()
