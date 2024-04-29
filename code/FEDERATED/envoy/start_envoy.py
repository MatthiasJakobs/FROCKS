import yaml
import sys
import os

# Ottieni l'argomento dalla riga di comando
dataset = sys.argv[1]
client = sys.argv[2]
rocket = bool(int(sys.argv[3]))
kernels = sys.argv[4]
myseed = sys.argv[5]
    
print("Dataset: ", dataset)
print("Client: ", client)
print("ROCKET: ", rocket)
print("If ROCKET true, kernels: ", kernels)
print("Seed: ", myseed)

class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

rank_worldsize = [client,4]
# Crea un dizionario con i dati necessari da includere nel file YAML
yaml_data = {
    'params': {
        'cuda_devices': []},
    'optional_plugin_components': {},
    'shard_descriptor':{
        'template': "ucr_shard_descriptor.UCRShardDescriptor",
        'params': {
            "data_folder": "ucr_data",
            "rank_worldsize": list(rank_worldsize),
            "data_dir": dataset,
            "rocket": rocket, 
            "kernels": kernels,
            "seed": myseed}
    }
}

print("Modifying envoy_config.yaml shapes")
with open(f'envoy_config{client}.yaml', 'r') as yaml_file:
    yaml_data = yaml.load(yaml_file, Loader=PrettySafeLoader)

# Modifica il file YAML con le dimensioni ottenute
yaml_data['shard_descriptor']['params']['data_dir'] = dataset
yaml_data['shard_descriptor']['params']['rocket'] = rocket
yaml_data['shard_descriptor']['params']['kernels'] = kernels
yaml_data['shard_descriptor']['params']['seed'] = myseed
#yaml_data['shard_descriptor']['params']['rank_worldsize'] = list(client + 2)

# Sovrascrivi il file YAML con i dati modificati
with open(f'envoy_config{client}.yaml', 'w') as yaml_file:
    yaml.dump(yaml_data, yaml_file, default_flow_style=False)

print(f"Starting envoy{client}...")
os.system(f"sh start_envoy{client}.sh {dataset}_env_{client}")



