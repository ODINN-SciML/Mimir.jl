module Mimir

using Lux
using Random
using CSV
using DataFrames
using JSON
using Infiltrator

export MLP, CustomMLP, infer_MLP_size, load_data, load_params_from_json

include("MLP.jl")
include("data_reader.jl")

end