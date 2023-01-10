import argparse
import json
import os
import itertools
import copy
import tempfile
import yaml
import pprint
import argparse

# Global configuration values for default output and storage.
repo_path = os.path.dirname(os.path.dirname(__file__))
STORAGE_ROOT = os.path.dirname(repo_path)
ENV_SETUP_SCRIPT = os.path.join(repo_path, "setup_shell.sh")
TMP_DIR = os.path.join(STORAGE_ROOT, "tmp")
DEFAULT_ENTRY_POINT = "scripts/train.py"
DEFAULT_REQUIRED_ARGS = ["path", "config"]

# Specifies which config values will split experiments into folders
# by default this is just the environment.
FOLDER_KEYS = ["env"]

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-point", type=str, action='append', default=None)
    parser.add_argument("--arguments", metavar="KEY=VALUE", nargs='+', action='append', help="Set kv pairs used as args for the entry point script.")
    parser.add_argument("--seeds-per-job", type=int, default=1)
    return parser

def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return (key, value)

def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d

def get_jobs(args):
    all_jobs = []

    if args.entry_point is None:
        # If entry point wasn't provided use the default
        args.entry_point = [DEFAULT_ENTRY_POINT]
    if len(args.entry_point) < len(args.arguments):
        # If we only were given one entry point but many script arguments, replicate the entry point
        assert len(args.entry_point) == 1
        args.entry_point = args.entry_point * len(args.arguments)

    for entry_point, arguments in zip(args.entry_point, args.arguments):
        script_args = parse_vars(arguments)
        # Handle the default case, train.
        if entry_point == DEFAULT_ENTRY_POINT:
            '''
            Custom code for sweeping using the experiment sweeper.
            '''
            for arg_name in DEFAULT_REQUIRED_ARGS:
                assert arg_name in script_args

            if script_args['config'].endswith(".json"):
                experiment = Experiment.load(script_args['config'])
                configs_and_paths = [(c, os.path.join(script_args['path'], n)) for c, n in experiment.generate_configs_and_names()]
            else:
                configs_and_paths = [(script_args['config'], script_args['path'])]

            jobs = [{"config": c, "path" : p} for c, p in configs_and_paths]
            for arg_name in script_args.keys():
                if not arg_name in jobs[0]:
                    print("Warning: argument", arg_name, "being added globally to all python calls with value", script_args[arg_name])
                    for job in jobs:
                        job[arg_name] = script_args[arg_name]
        else:
            # we have the default configuration. When there are multiple jobs per instance, 
            # We replicate the same job many times on the machine.
            jobs = [script_args]
        
        if args.seeds_per_job > 1:
            # copy all of the configratuions and add seeds
            seeded_jobs = []
            for job in jobs:
                seed = int(job.get('seed'))
                for i in range(args.seeds_per_job):
                    seeded_job = job.copy() # Should be a shallow dictionary, so copy OK
                    seeded_job['seed'] = seed + i
                    seeded_jobs.append(seeded_job)
            # Replace regular jobs with the seeded variants.
            jobs = seeded_jobs

        # add the entry point
        jobs = [(entry_point, job_args) for job_args in jobs]
        all_jobs.extend(jobs)

    return all_jobs

class Config(object):
    '''
    A lightweight copy of the config file with only basic IO capabilities.
    This is used so that we don't load in the full package on slurm head nodes.
    This is a bit of a work around for now, but it saves a lot of time.
    '''

    def __init__(self):
        # Define the necesary structure for a complete training configuration
        self.parsed = False
        self.config = dict()

    def save(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    def update(self, d):
        self.config.update(d)

    @classmethod
    def load(cls, path):
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.Loader)
        config = cls()
        config.update(data)
        return config

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return self.config.__contains__(key)

    def __str__(self):
        return pprint.pformat(self.config, indent=4)

    def copy(self):
        assert not self.parsed, "Cannot copy a parsed config"
        config = type(self)()
        config.config = copy.deepcopy(self.config)
        return config

class Experiment(dict):

    def __init__(self, base=None, name=None, paired_keys=[]):
        super().__init__()
        self._name = name
        self.base_config = Config.load(base)
        self.paired_keys = paired_keys

    @property
    def name(self):
        return self._name

    @classmethod
    def load(cls, path):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'r') as fp:
            data = json.load(fp)
        # Run formatting checks
        assert 'base' in data, "Did not supply a base config"
        base_config = data['base']
        del data['base'] # Remove the base configuration

        if 'paired_keys' in data:
            # We have some paired values. This means that in the variant updater these are all changed at the same time.
            paired_keys = data['paired_keys']
            assert isinstance(paired_keys, list)
            if len(paired_keys) > 0:
                assert all([isinstance(key_pair, list) for key_pair in paired_keys])
            del data['paired_keys']
        else:
            paired_keys = []

        for k, v in data.items():
            assert isinstance(k, str)
            assert isinstance(v, list)
        experiment = cls(base=base_config, name=name, paired_keys=paired_keys)
        experiment.update(data)
        return experiment

    def get_variants(self):
        paired_keys = set()
        for key_pair in self.paired_keys:
            for k in key_pair:
                if k in paired_keys:
                    raise ValueError("Key was paired multiple times!")
                paired_keys.add(k)
        
        groups = []
        unpaired_keys = [key for key in self.keys() if not key in paired_keys] # Fix the ordering!
        unpaired_variants = itertools.product(*[self[k] for k in unpaired_keys])
        unpaired_variants = [{key:variant[i] for i, key in enumerate(unpaired_keys)} for variant in unpaired_variants]
        groups.append(unpaired_variants)

        # Now construct the paired variants
        for key_pair in self.paired_keys:
            # instead of using product, use zip
            pair_variant = zip(*[self[k] for k in key_pair]) # This gets all the values
            pair_variant = [{key:variant[i] for i, key in enumerate(key_pair)} for variant in pair_variant]
            groups.append(pair_variant)

        group_variants = itertools.product(*groups)
        # Collapse the variants, making sure to copy the dictionaries so we don't get duplicates
        variants = []
        for variant in group_variants:
            collapsed_variant = {k:v for x in variant for k,v in x.items()}
            variants.append(collapsed_variant)

        return variants

    def generate_configs_and_names(self):
        variants = self.get_variants()
        configs_and_names = []
        for i, variant in enumerate(variants):
            config = self.base_config.copy()
            name = ""
            seed = None
            remove_trailing_underscore = False
            for k, v in variant.items():
                config_path = k.split('.')
                config_dict = config
                # Recursively update the current config until we find the value.
                while len(config_path) > 1:
                    if not config_path[0] in config_dict:
                        raise ValueError("Experiment specified key not in config: " + str(k))
                    config_dict = config_dict[config_path[0]]
                    config_path.pop(0)
                if not config_path[0] in config_dict:
                        raise ValueError("Experiment specified key not in config: " + str(k))
                # Finally set the value
                config_dict[config_path[0]] = v
                
                if k in FOLDER_KEYS and len(self[k]) > 1:
                    name = os.path.join(v, name)
                elif k == "seed" and len(self["seed"]) > 1: # More than one seed specified.
                    seed = v # Note that seed is not added to the name.
                elif len(self[k]) > 1:
                    # Add it to the path name if it is different for each run.
                    if isinstance(v, str):
                        str_val = v
                    elif isinstance(v, int) or isinstance(v, float) or isinstance(v, bool) or v is None:
                        str_val = str(v)
                    elif isinstance(v, list):
                        str_val = '_'.join([str(val) for val in v])                        
                    else:
                        raise ValueError("Could not convert config value to str.")

                    name += str(config_path[0]) + '-' + str_val + '_'
                    remove_trailing_underscore = True

            if remove_trailing_underscore:
                name = name[:-1]
            name = os.path.join(self.name, name)
            if seed is not None:
                name = os.path.join(name, "seed-" + str(seed))
            if not os.path.exists(TMP_DIR):
                os.mkdir(TMP_DIR)
            _, config_path = tempfile.mkstemp(text=True, prefix='config_', suffix='.json', dir=TMP_DIR)
            print("Variant", i+1)
            print(config)
            config.save(config_path)
            configs_and_names.append((config_path, name))
        
        return configs_and_names
