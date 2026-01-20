"""
    MLP(nNeurons::Vector, activation=relu)

A simple feedforward neural network built dynamically based on layer sizes.

# Arguments
- `nNeurons::Vector`: Vector of layer sizes, must have at least 2 elements.
  Example: [input_size, hidden_size_1, ..., hidden_size_n, output_size]
- `activation`: Activation function to use between layers (default: `relu`)
  Example: `relu`, `tanh`, `sigmoid`, `gelu`, etc.
"""
function MLP(nNeurons::Vector, activation=relu)
    @assert length(nNeurons) >= 2 "nNeurons must have at least 2 elements"
    
    layers = []
    
    # First layer
    push!(layers, Dense(nNeurons[1], nNeurons[2]))
    
    # Hidden layers with activation function
    for i in 2:length(nNeurons)-1
        push!(layers, activation)
        push!(layers, Dense(nNeurons[i], nNeurons[i+1]))
    end
    
    # Final output layer (linear, no activation)
    push!(layers, Dense(nNeurons[end], 1))
    
    return Chain(layers...)
end

"""
    CustomMLP

A custom neural network regressor struct that wraps a Lux model with training configuration.

# Fields
- `model`: The Lux neural network model
- `nbFeatures::Int`: Number of input features
- `nNeurons::Vector`: Layer sizes for the network
- `batch_size::Int`: Batch size for training (default: 16)
- `activation`: Activation function (default: relu)
- `device::String`: Device to run on, "cpu" or "gpu" (default: "cpu")
- `shuffle::Bool`: Whether to shuffle training data (default: true)
- `params::NamedTuple`: Model parameters
- `state::NamedTuple`: Model state
"""
struct CustomMLP
    model::Lux.AbstractLuxLayer
    nbFeatures::Int
    nNeurons::Vector
    batch_size::Int
    activation::Function
    device::String
    shuffle::Bool
    params::NamedTuple
    state::NamedTuple
end

"""
    CustomMLP(;
        nbFeatures::Int,
        nNeurons::Vector,
        batch_size::Int=16,
        activation=relu,
        device::String="cpu",
        shuffle::Bool=true
    )

Create a CustomMLP with the specified configuration.
"""
function CustomMLP(;
    nbFeatures::Int,
    nNeurons::Vector,
    batch_size::Int=16,
    activation=relu,
    device::String="cpu",
    shuffle::Bool=true
)
    # Ensure nbFeatures matches nNeurons first dimension
    @assert nNeurons[1] == nbFeatures "nNeurons[1] must equal nbFeatures"
    
    # Create the model
    model = MLP(nNeurons, activation)
    
    # Initialize parameters and state
    rng = Random.default_rng()
    params, state = Lux.setup(rng, model)
    
    return CustomMLP(
        model,
        nbFeatures,
        nNeurons,
        batch_size,
        activation,
        device,
        shuffle,
        params,
        state
    )
end

"""
    infer_MLP_size(weights::NamedTuple) -> Vector{Int}

Infer the MLP layer sizes from model weights.

Extracts the input and output dimensions from Dense layer weights to reconstruct
the network architecture.

# Arguments
- `weights::NamedTuple`: Model parameters containing layer weights

# Returns
- `Vector{Int}`: Layer sizes [input_size, hidden_sizes..., output_size]
"""
function infer_MLP_size(weights::NamedTuple)
    layer_sizes = []
    
    # Iterate through the weight structure
    for key in keys(weights)
        if haskey(weights[key], :weight)
            w = weights[key][:weight]
            # weight matrix shape is (out_features, in_features)
            out_features, in_features = size(w)
            
            # Add input size from first layer
            if isempty(layer_sizes)
                push!(layer_sizes, in_features)
            end
            
            # Add output size
            push!(layer_sizes, out_features)
        end
    end
    
    return layer_sizes
end