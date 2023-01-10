import tempfile
import subprocess
import os
import utils
import argparse
import copy

if __name__ == "__main__":

    parser = utils.get_parser()
    parser.add_argument('--cpus', '-c', type=str, nargs='+', default=None)
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=None)
    
    # Add Taskset and GPU arguments
    args = parser.parse_args()
    assert isinstance(args.gpus, list) or args.gpus is None, "GPUs must be a list of ints or None."
    assert isinstance(args.cpus, list) or args.cpus is None, "CPUs must be a list"
    
    jobs = utils.get_jobs(args)

    # Compute the total CPU Range
    if args.cpus is None:
        args.cpus = ['0-' + str(os.cpu_count())]
    cpu_list = []
    for cpu_item in args.cpus:
        if isinstance(cpu_item, str) and '-' in cpu_item:
            # We have a CPU range
            cpu_min, cpu_max = cpu_item.split('-')
            cpu_min, cpu_max = int(cpu_min), int(cpu_max)
            cpu_list.extend(list(range(cpu_min, cpu_max)))
        else:
            cpu_list.append(int(cpu_item))
    cores_per_job = len(cpu_list) // len(jobs)
    
    gpus = args.gpus
    
    processes = []
    for i, (entry_point, script_ags) in enumerate(jobs):
        job_cpus = ','.join([str(c) for c in cpu_list[i*cores_per_job:(i+1)*cores_per_job]])
        command_list = [
            'taskset', '-c', job_cpus,
            'python', entry_point
        ]
        for arg_name, arg_value in script_ags.items():
            command_list.append("--" + arg_name)
            command_list.append(str(arg_value))
        
        job_gpus = str(gpus[i % len(gpus)]) if gpus is not None else None
        if job_gpus is not None:
            env = os.environ
            env["CUDA_VISIBLE_DEVICES"] = job_gpus # TODO: this doesn't support multi-gpu
        else:
            env = None
        print("[Local Sweeper] Starting job on gpu:", job_gpus, "and cpus:", job_cpus)
        proc = subprocess.Popen(command_list, env=env)
        processes.append(proc)
        
    try:
        exit_codes = [p.wait() for p in processes]
        print("[Local Sweeper] Waiting for completion.")
    except KeyboardInterrupt:
        for p in processes:
            try:
                p.terminate()
            except OSError:
                pass
            p.wait()
