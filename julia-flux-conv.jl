### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ e8731000-0713-11eb-1aa5-85850514f933
begin
	using Flux
	using Flux.Data.MNIST
	using Flux:outdims
	using Images
	using Images:permuteddimsview
end

# ╔═╡ 4f0b4b60-0715-11eb-2b73-bf6a98ce8dd9
md"
## Loading the MNIST dataset
"

# ╔═╡ 2b818160-0714-11eb-03a8-012df9ac4ab7
begin
	training_images = MNIST.images()
	test_images = MNIST.images(:test)
	training_labels = MNIST.labels()
	test_labels = MNIST.labels(:test)
end

# ╔═╡ 38dc8980-0af3-11eb-0899-71aeaeb9f863
md"
### Data Specifications:
"

# ╔═╡ 2d895e00-0af3-11eb-2c77-e7d552a10586
md"
**Type:**
$(typeof(training_images[1]))
"

# ╔═╡ 4180e8b0-0af3-11eb-34ab-21d2b7f723ef
md"
**Size:**
$(size(training_images[1]))
"

# ╔═╡ 8d21def2-0af3-11eb-0f5f-fdf582ed2160
md"
### Sample Training Data and Labels
"

# ╔═╡ 4aa58040-0715-11eb-0603-4d736f5fcf7d
training_images[1:3]

# ╔═╡ bfdea380-0af0-11eb-0a85-e563b1ff38d3
training_labels[1:3]

# ╔═╡ d121c110-0715-11eb-29c0-57e56bc73dec
md"
## Building the model
From the model zoo: [simple ConvNets on MNIST](https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl)
"

# ╔═╡ 1249742e-07de-11eb-1c38-e5a1521141ff
begin
	image_channel = 1
	image_size = (28, 28, image_channel)
	output_channel = 32
end

# ╔═╡ ef78bc90-0af3-11eb-3618-2703383f79eb
md"
### Convolutional and Dense Layers
"

# ╔═╡ 1b97c760-0af1-11eb-36b6-53aaada821fc
md"
[Flux model reference](https://fluxml.ai/Flux.jl/stable/models/layers/#Convolution-and-Pooling-Layers-1):
Convolutional layers expect input of dimension (width, height, channel, batch size)
"

# ╔═╡ 487f79a2-07de-11eb-3bcb-45a30b8416f5
convolutional_layers = Chain(
	Conv((3,3), image_channel=>16, pad=(1,1), relu),
	MaxPool((2,2)),
	Conv((3,3), 16=>32, pad=(1,1), relu),
	MaxPool((2,2)),
	Conv((3,3), 32=>output_channel, pad=(1,1), relu),
	MaxPool((2,2))
)

# ╔═╡ b0c75642-07de-11eb-379d-a1692550fa02
begin
	output_dim = outdims(convolutional_layers, image_size)
	flattened_output_size = prod(output_dim) * output_channel
end

# ╔═╡ daa6e350-0715-11eb-02c4-9373a60ad880
fully_connected_layers = Chain(
	Dense(flattened_output_size, 10)
)

# ╔═╡ 36cd55a0-07df-11eb-2eb8-151890ffb20b
model = Chain(
	convolutional_layers,
	flatten,
	fully_connected_layers
)

# ╔═╡ 0292b882-0af4-11eb-35bf-4397d85d3717
md"
### Syntactic Check
"

# ╔═╡ 99f99980-0af1-11eb-3f55-53e60616b8d6
begin
	singleton_batch_size = 1
	singleton_batch = Array{Float32}(undef, 28, 28, 1, singleton_batch_size)
	singleton_batch[:, :, :, 1] = training_images[1]
	nothing
end

# ╔═╡ 21e8a432-0af2-11eb-1c85-072903d3e3c2
model(singleton_batch)

# ╔═╡ d2d66a10-0af3-11eb-2b1f-8b9edadb8556
md"
## Data Preprocessing
"

# ╔═╡ Cell order:
# ╠═e8731000-0713-11eb-1aa5-85850514f933
# ╟─4f0b4b60-0715-11eb-2b73-bf6a98ce8dd9
# ╠═2b818160-0714-11eb-03a8-012df9ac4ab7
# ╟─38dc8980-0af3-11eb-0899-71aeaeb9f863
# ╟─2d895e00-0af3-11eb-2c77-e7d552a10586
# ╟─4180e8b0-0af3-11eb-34ab-21d2b7f723ef
# ╟─8d21def2-0af3-11eb-0f5f-fdf582ed2160
# ╠═4aa58040-0715-11eb-0603-4d736f5fcf7d
# ╠═bfdea380-0af0-11eb-0a85-e563b1ff38d3
# ╟─d121c110-0715-11eb-29c0-57e56bc73dec
# ╠═1249742e-07de-11eb-1c38-e5a1521141ff
# ╟─ef78bc90-0af3-11eb-3618-2703383f79eb
# ╟─1b97c760-0af1-11eb-36b6-53aaada821fc
# ╠═487f79a2-07de-11eb-3bcb-45a30b8416f5
# ╠═b0c75642-07de-11eb-379d-a1692550fa02
# ╠═daa6e350-0715-11eb-02c4-9373a60ad880
# ╠═36cd55a0-07df-11eb-2eb8-151890ffb20b
# ╟─0292b882-0af4-11eb-35bf-4397d85d3717
# ╠═99f99980-0af1-11eb-3f55-53e60616b8d6
# ╠═21e8a432-0af2-11eb-1c85-072903d3e3c2
# ╟─d2d66a10-0af3-11eb-2b1f-8b9edadb8556
