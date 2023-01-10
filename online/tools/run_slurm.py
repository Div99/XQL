import tempfile
import subprocess
import os
import utils
import argparse
import copy

SLURM_LOG_DEFAULT = os.path.join(utils.STORAGE_ROOT, "slurm_logs")

SLURM_ARGS = {
    "partition" : {"type": str, "required": True},
    "time" : {"type": str, "default": "48:00:00"},
    "nodes": {"type": int, "default": 1},
    "ntasks-per-node": {"type": int, "default": 1},
    "cpus": {"type": int, "required": True},
    "gpus": {"type": str, "required": False, "default": None},
    "mem": {"type": str, "required": True},
    "output": {"type" : str, "default": SLURM_LOG_DEFAULT},
    "error": {"type" : str, "default": SLURM_LOG_DEFAULT},
    "job-name" : {"type": str, "required": True},
    "exclude" : {"type": str, "required": False, "default": None},
    "nodelist": {"type": str, "required": False, "default": None}
}

SLURM_NAME_OVERRIDES = {
    "gpus" : "gres",
    "cpus": "cpus-per-task"
}

def write_slurm_header(f, args):
    # Make a copy of the args to prevent corruption
    args = copy.deepcopy(args)
    # Modify everything in the name space to later write it all at once
    for key in SLURM_ARGS.keys():
        assert key.replace('-', '_') in args, "Key " + key + " not found."
    
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    if not os.path.isdir(args.error):
        os.makedirs(args.error)

    args.output = os.path.join(args.output, args.job_name + "_%A.out")
    args.error = os.path.join(args.error, args.job_name + "_%A.err")
    args.gpus = "gpu:" + str(args.gpus) if args.gpus is not None else args.gpus

    NL = '\n'
    f.write("#!/bin/bash" + NL)
    f.write(NL)
    for arg_name in SLURM_ARGS.keys():
        arg_value = vars(args)[arg_name.replace('-', '_')]
        if arg_name in SLURM_NAME_OVERRIDES:
            arg_name = SLURM_NAME_OVERRIDES[arg_name]
        if not arg_value is None:
            f.write("#SBATCH --" + arg_name + "=" + str(arg_value) + NL)
    
    f.write(NL)
    f.write('echo "SLURM_JOBID = "$SLURM_JOBID' + NL)
    f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST' + NL)
    f.write('echo "SLURM_JOB_NODELIST = "$SLURM_JOB_NODELIST' + NL)
    f.write('echo "SLURM_NNODES = "$SLURM_NNODES' + NL)
    f.write('echo "SLURMTMPDIR = "$SLURMTMPDIR' + NL)
    f.write('echo "working directory = "$SLURM_SUBMIT_DIR' + NL)
    f.write(NL)
    f.write(". " + utils.ENV_SETUP_SCRIPT)
    f.write(NL)

if __name__ == "__main__":

    parser = utils.get_parser()
    parser.add_argument("--jobs-per-instance", default=1, type=int, help="Number of jobs to run in one slurm sbatch.")
    # Add Slurm Arguments
    for k, v in SLURM_ARGS.items():
        parser.add_argument("--" + k, **v)

    args = parser.parse_args()
    jobs = utils.get_jobs(args)

    # Call python subprocess to launch the slurm jobs.
    num_slurm_calls = len(jobs) // args.jobs_per_instance
    remainder_jobs = len(jobs) - num_slurm_calls * args.jobs_per_instance
    jobs_per_call = [args.jobs_per_instance for _ in range(num_slurm_calls)]
    for i in range(remainder_jobs):
        jobs_per_call[i] += 1 # Add the remainder jobs to spread them out as evenly as possible.
    assert sum(jobs_per_call) == len(jobs)
    job_index = 0
    procs = []
    for num_jobs in jobs_per_call:
        current_jobs = jobs[job_index:job_index + num_jobs]
        job_index += num_jobs
        
        _, slurm_file = tempfile.mkstemp(text=True, prefix='job_', suffix='.sh')
        print("Launching job with slurm configuration:", slurm_file)

        with open(slurm_file, 'w+') as f:
            write_slurm_header(f, args)
            # Now that we have written the header we can launch the jobs.
            for entry_point, script_args in current_jobs:
                command_str = ['python', entry_point]
                for arg_name, arg_value in script_args.items():
                    command_str.append("--" + arg_name)
                    command_str.append(str(arg_value))
                if len(current_jobs) != 1:
                    command_str.append('&')
                command_str = ' '.join(command_str) + '\n'
                f.write(command_str)
            if len(current_jobs) != 1:
                f.write('wait')
            
            # Now launch the job
            proc = subprocess.Popen(['sbatch', slurm_file])
            procs.append(proc)

    exit_codes = [p.wait() for p in procs]

