import os
import yaml
import pprint
import importlib
import copy

class Config(object):

    def __init__(self):
        # Define the necesary structure for a complete training configuration
        self.parsed = False
        self.config = dict()

        # Env Args
        self.config['env'] = None
        self.config['env_kwargs'] = {}

        # Algorithm Args
        self.config['alg'] = None
        self.config['alg_kwargs'] = {}
        
        # Dataset args
        self.config['dataset'] = None
        self.config['dataset_kwargs'] = {}
        self.config['validation_dataset_kwargs'] = None

        # Dataloader arguments
        self.config['collate_fn'] = None
        self.config['batch_size'] = None

        # Processor arguments
        self.config['processor'] = None
        self.config['processor_kwargs'] = {}

        # Optimizer Args
        self.config['optim'] = None
        self.config['optim_kwargs'] = {}
        self.config['scheduler'] = None

        # network Args
        self.config['network'] = None
        self.config['network_kwargs'] = {}

        # Schedule args
        self.config['schedule'] = None
        self.config['schedule_kwargs'] = {}

        # General arguments
        self.config['checkpoint'] = None
        self.config['seed'] = None # Does nothing right now.
        self.config['train_kwargs'] = {}

    @staticmethod
    def _parse_helper(d):
        for k, v in d.items():
            if isinstance(v, list) and len(v) > 1 and v[0] == "import":
                # parse the value to an import
                d[k] = getattr(importlib.import_module(v[1]), v[2])
            elif isinstance(v, dict):
                Config._parse_helper(v)

    def parse(self):
        config = self.copy()
        Config._parse_helper(config.config)
        return config
        
    def update(self, d):
        self.config.update(d)
    
    def save(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, 'w') as f:
            yaml.dump(self.config, f)

    @classmethod
    def load(cls, path):
        if os.path.isdir(path):
            path = os.path.join(path, "config.yaml")
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.Loader)
        config = cls()
        config.update(data)
        return config

    @staticmethod
    def _flatten_helper(flattened_config, value, prefix, separator="."):
        if isinstance(value, dict) and all([isinstance(k, str) for k in value.keys()]):
            # We have another nested configuration dictionary
            for k in value.keys():
                Config._flatten_helper(flattened_config, value[k], prefix + separator + k, separator=separator)
        else:
            # We do not have a config file, just return the regular value.
            flattened_config[prefix[1:]] = value # Note that we remove the first prefix because it has a leading '.'
    
    def flatten(self, separator="."):
        '''Returns a flattened version of the config where '.' separates nested values'''
        flattened_config = {}
        Config._flatten_helper(flattened_config, self.config, "", separator=separator)
        return flattened_config

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return self.config.__contains__(key)

    def __str__(self):
        return pprint.pformat(self.config, indent=4)

    def copy(self):
        config = type(self)()
        config.config = copy.deepcopy(self.config)
        return config