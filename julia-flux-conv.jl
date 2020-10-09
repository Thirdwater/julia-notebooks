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

# ╔═╡ 4aa58040-0715-11eb-0603-4d736f5fcf7d
[training_images[1:3], training_labels[1:3]]

# ╔═╡ d121c110-0715-11eb-29c0-57e56bc73dec
md"
## Building the model
From the model zoo: [simple ConvNets on MNIST](https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl)
"

# ╔═╡ 1249742e-07de-11eb-1c38-e5a1521141ff
begin
	image_channel = 1
	image_size = (28, 28, image_channel)
end

# ╔═╡ 487f79a2-07de-11eb-3bcb-45a30b8416f5
convolutional_layers = Chain(
	Conv((3,3), image_channel=>16, pad=(1,1), relu),
	MaxPool((2,2)),
	Conv((3,3), 16=>32, pad=(1,1), relu),
	MaxPool((2,2)),
	Conv((3,3), 32=>32, pad=(1,1), relu),
	MaxPool((2,2))
)

# ╔═╡ b0c75642-07de-11eb-379d-a1692550fa02
convolutional_output_dim = outdims(convolutional_layers, image_size)

# ╔═╡ daa6e350-0715-11eb-02c4-9373a60ad880
fully_connected_layers = Chain(
	Dense(prod(convolutional_output_dim), 10)
)

# ╔═╡ 36cd55a0-07df-11eb-2eb8-151890ffb20b
model = Chain(
	convolutional_layers,
	flatten,
	fully_connected_layers
)

# ╔═╡ 4dca0a00-07df-11eb-3644-3d1a74281187
channelview(training_images[1])

# ╔═╡ 68bb6d40-0a46-11eb-1335-c166c9146fa5
reshape(channelview(training_images[1]), 28, 28, 1)

# ╔═╡ 0724f4ac-0a41-11eb-3ba1-ffa3b881d7a4
convolutional_layers(reshape(channelview(training_images[1]), 28, 28, 1))

# ╔═╡ Cell order:
# ╠═e8731000-0713-11eb-1aa5-85850514f933
# ╟─4f0b4b60-0715-11eb-2b73-bf6a98ce8dd9
# ╠═2b818160-0714-11eb-03a8-012df9ac4ab7
# ╟─4aa58040-0715-11eb-0603-4d736f5fcf7d
# ╟─d121c110-0715-11eb-29c0-57e56bc73dec
# ╠═1249742e-07de-11eb-1c38-e5a1521141ff
# ╠═487f79a2-07de-11eb-3bcb-45a30b8416f5
# ╟─b0c75642-07de-11eb-379d-a1692550fa02
# ╟─daa6e350-0715-11eb-02c4-9373a60ad880
# ╠═36cd55a0-07df-11eb-2eb8-151890ffb20b
# ╠═4dca0a00-07df-11eb-3644-3d1a74281187
# ╠═68bb6d40-0a46-11eb-1335-c166c9146fa5
# ╠═0724f4ac-0a41-11eb-3ba1-ffa3b881d7a4
