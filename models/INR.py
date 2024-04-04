import jax.numpy as jnp
import jax
import haiku as hk
import math
import numpy as np
from learned_optimization.research.general_lopt import prefab
import optax
from utils import others

class INIF_model(hk.Module):
    def __init__(self, features: int,
                 coords_channel: int, data_channel: int = 1,
                 layers: int = 7, frequency: float = 90, **kwargs):
        super().__init__()
        self.features = features
        self.coords_channel = coords_channel
        self.data_channel = data_channel
        self.layers = layers
        self.frequency = frequency
        self.freq_adjust = hk.get_parameter("freq_adjust", shape=[1], init=hk.initializers.Constant(0))
        self.sensory = hk.Linear(self.features,
                                 name="sensory",
                                 w_init=hk.initializers.RandomUniform(
                                     minval=-1/self.coords_channel,
                                     maxval=1/self.coords_channel),
                                 )
        for i in range(1, self.layers-1):  # intermediate layers = layers-2
            self.__setattr__(f"mid_{i}",
                         hk.Linear(self.features,
                                   name=f"mid_{i}",
                                   w_init=hk.initializers.RandomUniform(
                                       minval=-jnp.sqrt(6 / self.features).item() / self.frequency,
                                    maxval=jnp.sqrt(6 / self.features).item() / self.frequency),
                                   ))
        self.out = hk.Linear(self.data_channel, name="out")

    def __call__(self, coords):
        output = self.sensory(coords)
        output = jnp.sin(output * (self.frequency + self.freq_adjust))
        for i in range(1, self.layers - 1):  # intermediate layers = layers-2
            output = getattr(self, f"mid_{i}")(output)
            output = jnp.sin(output * (self.frequency + self.freq_adjust))
        return self.out(output).squeeze()

def INIF(features: int,
        coords_channel: int, data_channel: int = 1,
        layers: int = 7, frequency: float = 90, **kwargs):
    '''
    INIF model class
    
    Input -> coords: the coordinates of the data
    
    Output -> the predicted data
    
    Args:
    features: the number of features in the model
    coords_channel: the number of channels in the coordinates
    data_channel: the number of channels in the data
    layers: the number of layers in the model
    frequency: the frequency of the sine activation function
    '''
    @hk.transform
    def inner(coords):
        return INIF_model(features, coords_channel, data_channel,
                 layers, frequency, **kwargs)(coords)
    return inner

class SIREN_model(hk.Module):
    def __init__(self, features: int,
                 coords_channel: int, data_channel: int = 1,
                 layers: int = 7, frequency: float = 30, **kwargs):
        super().__init__()
        self.features = features
        self.coords_channel = coords_channel
        self.data_channel = data_channel
        self.layers = layers
        self.frequency = frequency
        self.sensory = hk.Linear(self.features,
                                 name="sensory",
                                 w_init=hk.initializers.RandomUniform(
                                     minval=-1/self.coords_channel,
                                     maxval=1/self.coords_channel),
                                 )
        for i in range(1, self.layers-1):  # intermediate layers = layers-2
            self.__setattr__(f"mid_{i}",
                         hk.Linear(self.features,
                                   name=f"mid_{i}",
                                   w_init=hk.initializers.RandomUniform(
                                       minval=-jnp.sqrt(6 / self.features).item() / self.frequency,
                                    maxval=jnp.sqrt(6 / self.features).item() / self.frequency),
                                   ))
        self.out = hk.Linear(self.data_channel, name="out")

    def __call__(self, coords):
        output = self.sensory(coords)
        output = jnp.sin(output * self.frequency)
        for i in range(1, self.layers - 1):  # intermediate layers = layers-2
            output = getattr(self, f"mid_{i}")(output)
            output = jnp.sin(output * self.frequency)
        return self.out(output).squeeze()

def SIREN(features: int,
        coords_channel: int, data_channel: int = 1,
        layers: int = 7, frequency: float = 30, **kwargs):
    '''
    SIREN model class
    
    Input -> coords: the coordinates of the data
    
    Output -> the predicted data
    
    Args:
    features: the number of features in the model
    coords_channel: the number of channels in the coordinates
    data_channel: the number of channels in the data
    layers: the number of layers in the model
    frequency: the frequency of the sine activation function
    '''
    @hk.transform
    def inner(coords):
        return SIREN_model(features, coords_channel, data_channel,
                 layers, frequency, **kwargs)(coords)
    return inner


def calc_features(param_count, coords_channel, data_channel=1, layers=7, **kwargs):
    '''
    this function calculates the number of features in the model
    
    input -> param_count: the number of parameters in the model
            coords_channel: the number of channels in the coordinates
            data_channel: the number of channels in the data
            layers: the number of layers in the model (default is 7)
            
    output -> features: the number of features in the model
    '''
    a = layers - 2
    b = coords_channel + 1 + layers - 2 + data_channel
    c = -param_count + data_channel
    if a == 0:
        features = round(-c / b)
    else:
        features = round((-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a))
    return features



def get_model_param(meta_data) -> tuple:
    '''
    this function initializes the model, parameters
    
    input -> meta_data: the metadata of the data
            sampler: the sampler object use for extracting a batch of data for initialization the model
    
    output -> model: the initialized model
            params: the initialized parameters
    '''
    raw_size = others.bytes_to_MB(meta_data['size'])
    target_param_count = round(meta_data['size'] / 4 / meta_data['ratio'])
    coord_dim = len(meta_data['shape'])
    coords = jnp.zeros(shape = (1, coord_dim), dtype=jnp.float32)
    target_features = calc_features(target_param_count,
                                    coords_channel=coord_dim,
                                    )
    
    seed = np.random.randint(0, 1000)
    key = jax.random.PRNGKey(seed)
    if meta_data['model']['name'] == 'SIREN':
        model = SIREN(features=target_features, coords_channel=coord_dim)
    elif meta_data['model']['name'] == 'INIF':
        model = INIF(features=target_features, coords_channel=coord_dim)
    elif meta_data['model']['name'] == 'HEVC':
        pass
    else:
        raise ValueError(f"Model name {meta_data['model']['name']} is not supported")
    params = model.init(key, coords)
    params_count = hk.data_structures.tree_size(params)
    meta_data['model']['features'] =  target_features
    meta_data['model']['coords_channel'] = coord_dim
    meta_data['seed'] = seed
    meta_data['key'] = key
    print(f"raw size: {raw_size}MB")
    print(f"target size {raw_size/meta_data['ratio']}MB, param count: {target_param_count}, ratio: {meta_data['ratio']}")
    print(f"actual param count: {params_count}, hidden size: {target_features}, ratio: {round((params_count/target_param_count)*meta_data['ratio'])}")
    return model, params

def create_learning_rate_schedule(total_steps,
                                  base,
                                  decay_type,
                                  warmup_steps,
                                  linear_end=1e-8):
  def step_fn(step):
    """Step to learning rate function."""
    lr = base

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = jnp.clip(progress, 0.0, 1.0)
    if decay_type == 'linear':
      lr = linear_end + (lr - linear_end) * (1.0 - progress)
    elif decay_type == 'cosine':
      lr = lr * 0.5 * (1. + jnp.cos(jnp.pi * progress))
    else:
      raise ValueError(f'Unknown lr type {decay_type}')

    if warmup_steps:
      lr = lr * jnp.minimum(1., step / warmup_steps)

    return jnp.asarray(lr, dtype=jnp.float32)

  return step_fn

def get_opt_opt_state(meta_data, params) -> tuple:
    '''
    this function initializes the optimizer and optimizer state
    
    input -> meta_data: the metadata of the data
            params: the initialized parameters
            
    output -> opt: the optimizer
            opt_state: the optimizer state
    '''
    iteration_number = meta_data['iteration_number']
    print(f"iteration number: {iteration_number}")
    meta_data['opt'] = {}
    if meta_data['model']['name'] == 'SIREN':
        meta_data['opt']['name'] = input("Enter the optimizer name(supports Adam, Adamax, AdamW... defualt Adamax): " or 'Adamax')
        meta_data['opt']['learning_rate'] = float(input("Enter the learning rate (default 2e-3): ") or 2e-3)
        lr_schedule = create_learning_rate_schedule(iteration_number, meta_data['opt']['learning_rate'], 'cosine', 0)
        if meta_data['opt']['name'] == 'Adam':
            opt = optax.chain(optax.clip_by_global_norm(1), optax.adam(lr_schedule))
        elif meta_data['opt']['name'] == 'AdamW':
            opt = optax.chain(optax.clip_by_global_norm(1), optax.adamw(lr_schedule))
        elif meta_data['opt']['name'] == 'SGD':
            opt = optax.chain(optax.clip_by_global_norm(1), optax.sgd(lr_schedule))
        else:
            opt = optax.chain(optax.clip_by_global_norm(1), optax.adamax(lr_schedule))
        opt_state = opt.init(params)
        
    else:
        meta_data['opt']['name'] = 'Lopt'
        opt = prefab.optax_lopt(iteration_number)
        opt_state = opt.init(params)
    
    return opt, opt_state

def get_model_param_opt_optstate(meta_data) -> tuple:
    '''
    this function initializes the model, parameters, optimizer, and optimizer state
    
    input -> meta_data: the metadata of the data
            sampler: the sampler object use for extracting a batch of data for initialization the model
    
    output -> model: the initialized model
            params: the initialized parameters
            opt: the optimizer
            opt_state: the optimizer state
    '''
    model, params = get_model_param(meta_data)
    opt, opt_state = get_opt_opt_state(meta_data, params)
    return model, params, opt, opt_state


