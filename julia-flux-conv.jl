### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ e8731000-0713-11eb-1aa5-85850514f933
begin
	using Flux
	using Flux.Data.MNIST
	using Flux: outdims
	using Flux: onehotbatch
	using Flux: onecold
	using Flux: logitcrossentropy
	using Statistics
	using BSON
	using BSON: @load
	
	using Base.Iterators: partition
	
	using PlutoUI
end

# ╔═╡ 8f4281c0-0b03-11eb-3d41-352461503c50
md"
# Julia and Flux Exercise on MNIST
"

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
	md"Loading data:"
end

# ╔═╡ 38dc8980-0af3-11eb-0899-71aeaeb9f863
md"
### Data Specifications:
"

# ╔═╡ 2d895e00-0af3-11eb-2c77-e7d552a10586
typeof(training_images[1])

# ╔═╡ 4180e8b0-0af3-11eb-34ab-21d2b7f723ef
size(training_images[1])

# ╔═╡ 8d21def2-0af3-11eb-0f5f-fdf582ed2160
md"
### Example Images and Labels
"

# ╔═╡ 4aa58040-0715-11eb-0603-4d736f5fcf7d
training_images[1:3]

# ╔═╡ bfdea380-0af0-11eb-0a85-e563b1ff38d3
training_labels[1:3]

# ╔═╡ 2319a4f0-0b04-11eb-0807-afd438471d7b
test_images[1:3]

# ╔═╡ 277d2620-0b04-11eb-1e39-633a07fafe32
test_labels[1:3]

# ╔═╡ d121c110-0715-11eb-29c0-57e56bc73dec
md"
## Building the model
From the model zoo: [simple ConvNets on MNIST](https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl)
"

# ╔═╡ 1249742e-07de-11eb-1c38-e5a1521141ff
begin
	image_size = (28, 28)
	image_channel = 1
	output_channel = 128
	md"Model input/output:"
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
function build_convolutional_layers()
	return Chain(
		Conv((3,3), image_channel=>16, pad=(1,1), relu),
		Conv((3,3), 16=>32, pad=(1,1), relu),
		Conv((3,3), 32=>64, pad=(1,1), relu),
		MaxPool((2,2)),
		Conv((5,5), 64=>output_channel, pad=(2,2), relu),
		MaxPool((2,2))
	)
end

# ╔═╡ b0c75642-07de-11eb-379d-a1692550fa02
begin
	output_dim = outdims(build_convolutional_layers(), image_size)
	flattened_output_size = prod(output_dim) * output_channel
	md"Calculating number of nodes at the end of the convolutional layers:"
end

# ╔═╡ 33afbe80-0d5c-11eb-2f86-fbcd9dd2b02b
[output_dim, flattened_output_size]

# ╔═╡ daa6e350-0715-11eb-02c4-9373a60ad880
function build_fully_connected_layers()
	return Chain(
		Dense(flattened_output_size, 32),
		Dense(32, 10)
	)
end

# ╔═╡ 197b65e2-0b87-11eb-236d-8d7d232c8b05
function build_model()
	return Chain(
		build_convolutional_layers(),
		flatten,
		build_fully_connected_layers(),
		softmax
	)
end

# ╔═╡ 36cd55a0-07df-11eb-2eb8-151890ffb20b
model = build_model()

# ╔═╡ 0292b882-0af4-11eb-35bf-4397d85d3717
md"
### Syntactic Check
"

# ╔═╡ 99f99980-0af1-11eb-3f55-53e60616b8d6
begin
	singleton_batch_size = 1
	singleton_batch = Array{Float32}(undef, 28, 28, 1, singleton_batch_size)
	singleton_batch[:, :, :, 1] = Float32.(training_images[1])
	md"Verifying the model syntax with input of batch size 1:"
end

# ╔═╡ 21e8a432-0af2-11eb-1c85-072903d3e3c2
model(singleton_batch)

# ╔═╡ d2d66a10-0af3-11eb-2b1f-8b9edadb8556
md"
## Data Preprocessing
"

# ╔═╡ c2a95f70-0af4-11eb-3d5f-19091e20b706
begin
	batch_size = 128
	batch_indices = partition(1:length(training_images), batch_size)
	md"Partitioning data into batches:"
end

# ╔═╡ 24fa6fb0-0b00-11eb-26b5-67abc6637c9b
function preprocess_batch(images, labels, indices)
	batch_images = Array{Float32}(
		undef, image_size..., image_channel, length(indices))
	for i in 1:length(indices)
		batch_images[:, :, :, i] = Float32.(images[indices[i]])
	end
	batch_labels = onehotbatch(labels[indices], 0:9)
	return (batch_images, batch_labels)
end

# ╔═╡ 6d07d790-0b03-11eb-11ab-45069658b240
md"
### Training Set
"

# ╔═╡ dd197cc0-0b01-11eb-14bd-df7b129e76cb
begin
	training_set = map(
		indices -> preprocess_batch(training_images, training_labels, indices),
		batch_indices
	)
	md"Preprocess training images and labels into batches:"
end

# ╔═╡ 1ce577e0-0b03-11eb-201b-4dd03b8aba2d
typeof(training_set)

# ╔═╡ 3a0d9c30-0b03-11eb-336a-a914eafc6d57
length(training_set)

# ╔═╡ c1cc2b90-0b04-11eb-017d-ef29a86a7185
size(training_set[1][1])

# ╔═╡ ca0d3510-0b04-11eb-24e9-236d2390f0db
size(training_set[1][2])

# ╔═╡ 783e5350-0b03-11eb-30b3-dd0b4340a7a8
md"
### Testing Set
"

# ╔═╡ 87d4a6c0-0b03-11eb-3e3b-49b763c0d3f8
begin
	testing_set = [
		preprocess_batch(test_images, test_labels, 1:length(test_images))
	]
	md"Treat the whole testing set as one batch:"
end

# ╔═╡ d768dac0-0b04-11eb-1e94-d71b61b80d2a
typeof(testing_set)

# ╔═╡ 9a260610-0b04-11eb-1173-9d173e66f823
length(testing_set)

# ╔═╡ 7082ff20-0b04-11eb-1d26-492330fad3fc
size(testing_set[1][1])

# ╔═╡ d140780e-0b04-11eb-35ff-97dd61791caa
size(testing_set[1][2])

# ╔═╡ a1497520-0b05-11eb-2c08-656810b1cfc5
md"
### Syntactic Check
"

# ╔═╡ 94972ed2-0b05-11eb-38b5-3d62be44f33f
model(training_set[1][1])

# ╔═╡ 8ddc33a0-0b06-11eb-391b-b39bd91e1ed0
md"
## Training
"

# ╔═╡ 7133b4b0-0b08-11eb-307e-e973adf1d112
md"
### Setup
"

# ╔═╡ a00a6ffe-0b06-11eb-095b-5f0b6bb8ad74
begin
	learning_rate = 0.00001
	optimiser = ADAM(learning_rate)
end

# ╔═╡ ee0ed1b0-0b06-11eb-380a-450c083feac1
function logit_crossentropy_loss(data, ground_truth)
	estimation = model(data)
	return logitcrossentropy(estimation, ground_truth)
end

# ╔═╡ 8818dee2-0b07-11eb-32cf-29e2680def32
logit_crossentropy_loss(training_set[1][1], training_set[1][2])

# ╔═╡ 4a8f8b50-0b07-11eb-22d2-2fff43eea24c
function accuracy(estimator, data, ground_truth)
	estimation = estimator(data)
	return mean( onecold(estimation) .== onecold(ground_truth) )
end

# ╔═╡ b09bcfd2-0b07-11eb-3e06-07a3ed3d5ff5
md"
Expect randomly initialised model to perform at around 10%:

$(accuracy(build_model(), testing_set[1][1], testing_set[1][2]))
"

# ╔═╡ eaaafb50-0b08-11eb-1b5a-c7c7b50a6b34
md"
### Training Loop
"

# ╔═╡ d97bcd80-0b0a-11eb-370c-df72b16367c0
begin
	max_epochs = 500
	max_accuracy = 0.99
	max_stalling_epochs = 50
	md"Exit conditions:"
end

# ╔═╡ f2222520-0b08-11eb-24f0-ff34ac9a64ad
function train()
	@info("Start training...")
	@info("\tImage size:\t$(image_size)")
	@info("\tTraining data:\t$(length(training_images))")
	@info("\tTesting data:\t$(length(test_images))")
	@info("\tBatch size:\t$(batch_size)")
	@info("\tNum batches:\t$(length(training_set))")
	@info("\tMax epochs:\t$(max_epochs)")
	@info("\tTarget acc:\t$(max_accuracy)")
	
	still_training = true
	epoch = 1
	best_accuracy = 0
	test_accuracy = 0
	last_improving_epoch = 1
	loss = (x, y) -> logit_crossentropy_loss(x, y)
	
	while still_training
		@info("  Epoch $(epoch)")
		Flux.train!(loss, Flux.params(model), training_set, optimiser)
		
		test_accuracy = accuracy(model, testing_set[1]...)
		@info("    Test Accuracy:\t$(test_accuracy)")
		
		if test_accuracy > best_accuracy
			best_accuracy = test_accuracy
			last_improving_epoch = epoch
			@info("    Saving to model2_epoch@$(epoch)_acc$(test_accuracy).bson")
			bson("./model2_epoch@$(epoch)_acc$(test_accuracy).bson",
				model=model,
				params=Flux.params(model),
				epochs=epoch,
				accuracy=test_accuracy
			)
		end
		
		epoch = epoch + 1
		if epoch > max_epochs
			@info("Max epoch reached!")
			still_training = false
		end
		if best_accuracy >= max_accuracy
			@info("Target accuracy reached!")
			still_training = false
		end
		if epoch - last_improving_epoch >= max_stalling_epochs
			@info("Max number of non-improving epochs reached!")
			still_training = false
		end
	end
	
	bson("./model2_epoch@$(epoch)_acc$(test_accuracy).bson",
		model=model,
		params=Flux.params(model),
		epochs=epoch,
		accuracy=test_accuracy
	)
	
	@info("Done!")
end

# ╔═╡ 57737e10-0b09-11eb-0141-739c8c5dad31
md"
Train:
$(@bind doTraining CheckBox())
"

# ╔═╡ 9e854f40-0b09-11eb-3b45-3b271d803194
if doTraining
	train()
end

# ╔═╡ 2b2e42a0-0b85-11eb-0bde-9bcd392dad49
md"
Load:
$(@bind load_model CheckBox())
"

# ╔═╡ 65de17c0-0b82-11eb-1d9f-7b21be10fa44
if load_model
	Core.eval(Main, :(import Flux))
	Core.eval(Main, :(import Zygote))
	Core.eval(Main, :(import NNlib))
	md"See [issue with loading BSON in a module](https://github.com/FluxML/Flux.jl/issues/1322):"
end

# ╔═╡ 0a7a3540-0b81-11eb-19b7-c31035f7900d
if load_model
	model_dict = BSON.load("./model2_epoch@35_acc0.9797.bson")
	typeof(model_dict)
end

# ╔═╡ 50fdfa70-0b85-11eb-075d-63196ba1fe72
if load_model
	model2 = build_model()
	Flux.loadparams!(model2, model_dict[:params])
	model2(training_set[1][1])
end

# ╔═╡ 0bfdd280-0c46-11eb-357f-69548db27f4d
if load_model
	Flux.loadparams!(model, model_dict[:params])
end

# ╔═╡ 767bc900-0c46-11eb-253b-e375a05b9707
if load_model
	model_dict[:epochs]
end

# ╔═╡ 7c29cb90-0c46-11eb-1bd1-3731b7c47830
if load_model
	model_dict[:accuracy]
end

# ╔═╡ 4ed3cec0-0c46-11eb-298c-91d26e1b794c
accuracy(model, testing_set[1][1], testing_set[1][2])

# ╔═╡ 7fd762f0-0b85-11eb-1372-61d1c1636e70
if load_model
	accuracy(model2, testing_set[1][1], testing_set[1][2])
end

# ╔═╡ 45e07300-0b87-11eb-11bd-4f31cd4ebde5
if load_model
	accuracy(build_model(), testing_set[1][1], testing_set[1][2])
end

# ╔═╡ Cell order:
# ╟─8f4281c0-0b03-11eb-3d41-352461503c50
# ╠═e8731000-0713-11eb-1aa5-85850514f933
# ╟─4f0b4b60-0715-11eb-2b73-bf6a98ce8dd9
# ╠═2b818160-0714-11eb-03a8-012df9ac4ab7
# ╟─38dc8980-0af3-11eb-0899-71aeaeb9f863
# ╠═2d895e00-0af3-11eb-2c77-e7d552a10586
# ╠═4180e8b0-0af3-11eb-34ab-21d2b7f723ef
# ╟─8d21def2-0af3-11eb-0f5f-fdf582ed2160
# ╠═4aa58040-0715-11eb-0603-4d736f5fcf7d
# ╠═bfdea380-0af0-11eb-0a85-e563b1ff38d3
# ╠═2319a4f0-0b04-11eb-0807-afd438471d7b
# ╠═277d2620-0b04-11eb-1e39-633a07fafe32
# ╟─d121c110-0715-11eb-29c0-57e56bc73dec
# ╠═1249742e-07de-11eb-1c38-e5a1521141ff
# ╟─ef78bc90-0af3-11eb-3618-2703383f79eb
# ╟─1b97c760-0af1-11eb-36b6-53aaada821fc
# ╠═487f79a2-07de-11eb-3bcb-45a30b8416f5
# ╠═b0c75642-07de-11eb-379d-a1692550fa02
# ╠═33afbe80-0d5c-11eb-2f86-fbcd9dd2b02b
# ╠═daa6e350-0715-11eb-02c4-9373a60ad880
# ╠═197b65e2-0b87-11eb-236d-8d7d232c8b05
# ╠═36cd55a0-07df-11eb-2eb8-151890ffb20b
# ╟─0292b882-0af4-11eb-35bf-4397d85d3717
# ╠═99f99980-0af1-11eb-3f55-53e60616b8d6
# ╠═21e8a432-0af2-11eb-1c85-072903d3e3c2
# ╟─d2d66a10-0af3-11eb-2b1f-8b9edadb8556
# ╠═c2a95f70-0af4-11eb-3d5f-19091e20b706
# ╠═24fa6fb0-0b00-11eb-26b5-67abc6637c9b
# ╟─6d07d790-0b03-11eb-11ab-45069658b240
# ╠═dd197cc0-0b01-11eb-14bd-df7b129e76cb
# ╠═1ce577e0-0b03-11eb-201b-4dd03b8aba2d
# ╠═3a0d9c30-0b03-11eb-336a-a914eafc6d57
# ╠═c1cc2b90-0b04-11eb-017d-ef29a86a7185
# ╠═ca0d3510-0b04-11eb-24e9-236d2390f0db
# ╟─783e5350-0b03-11eb-30b3-dd0b4340a7a8
# ╠═87d4a6c0-0b03-11eb-3e3b-49b763c0d3f8
# ╠═d768dac0-0b04-11eb-1e94-d71b61b80d2a
# ╠═9a260610-0b04-11eb-1173-9d173e66f823
# ╠═7082ff20-0b04-11eb-1d26-492330fad3fc
# ╠═d140780e-0b04-11eb-35ff-97dd61791caa
# ╟─a1497520-0b05-11eb-2c08-656810b1cfc5
# ╠═94972ed2-0b05-11eb-38b5-3d62be44f33f
# ╟─8ddc33a0-0b06-11eb-391b-b39bd91e1ed0
# ╟─7133b4b0-0b08-11eb-307e-e973adf1d112
# ╠═a00a6ffe-0b06-11eb-095b-5f0b6bb8ad74
# ╠═ee0ed1b0-0b06-11eb-380a-450c083feac1
# ╠═8818dee2-0b07-11eb-32cf-29e2680def32
# ╠═4a8f8b50-0b07-11eb-22d2-2fff43eea24c
# ╠═b09bcfd2-0b07-11eb-3e06-07a3ed3d5ff5
# ╟─eaaafb50-0b08-11eb-1b5a-c7c7b50a6b34
# ╠═d97bcd80-0b0a-11eb-370c-df72b16367c0
# ╠═f2222520-0b08-11eb-24f0-ff34ac9a64ad
# ╟─57737e10-0b09-11eb-0141-739c8c5dad31
# ╠═9e854f40-0b09-11eb-3b45-3b271d803194
# ╟─2b2e42a0-0b85-11eb-0bde-9bcd392dad49
# ╠═65de17c0-0b82-11eb-1d9f-7b21be10fa44
# ╠═0a7a3540-0b81-11eb-19b7-c31035f7900d
# ╠═50fdfa70-0b85-11eb-075d-63196ba1fe72
# ╠═0bfdd280-0c46-11eb-357f-69548db27f4d
# ╠═767bc900-0c46-11eb-253b-e375a05b9707
# ╠═7c29cb90-0c46-11eb-1bd1-3731b7c47830
# ╠═4ed3cec0-0c46-11eb-298c-91d26e1b794c
# ╠═7fd762f0-0b85-11eb-1372-61d1c1636e70
# ╠═45e07300-0b87-11eb-11bd-4f31cd4ebde5
