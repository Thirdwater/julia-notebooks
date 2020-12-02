### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ a77091e0-331e-11eb-09fc-f947365a92b6
begin
	using Flux: params
	using Flux: gradient
end

# ╔═╡ 4fb21b40-331e-11eb-3dd2-fdd16a08c245
md"
# Deep Learning with Flux Tutorial
<https://fluxml.ai/tutorials/2020/09/15/deep-learning-flux.html>
"

# ╔═╡ e8829700-331e-11eb-0f29-2dcab2f31f9a
md"
## Automatic Differentiation
"

# ╔═╡ ef8a4a30-33ef-11eb-1717-338b95c94bce
md"
`gradient` function takes a function and a set of arguments and returns the gradient of the input function wrt each arguments (see [docs](https://fluxml.ai/Flux.jl/stable/models/basics/#Taking-Gradients-1)).
The gradient can then be used to update the parameters accordingly (e.g. via gradient descent).
"

# ╔═╡ f08278d0-331e-11eb-280d-cf00798786a3
function taylor_sine(x)
	sine = 0
	for k in 0:8
		sign = (-1)^k
		numerator = x^(2k + 1)
		denominator = factorial(2k + 1)
		sine += sign * numerator / denominator
	end
	return sine
end

# ╔═╡ d679c780-331f-11eb-0421-d56b7f761d15
s = 0.5

# ╔═╡ b08808c0-331f-11eb-1901-a1ce85ac5400
taylor_sine(s), sin(s)

# ╔═╡ e64b4f30-331f-11eb-1282-518125edafd7
gradient(taylor_sine, s), cos(s)

# ╔═╡ 2a2e4ea0-3320-11eb-1091-733f1268d24f
md"
### Example on Traditional Neural Networks
"

# ╔═╡ 40dd3120-3320-11eb-158a-ef5f9c7a54d2
begin
	local W = randn(3, 5)
	local b = zeros(3)
	local x = rand(5)
	
	function loss(W, b, x)
		return sum(W * x .+ b)
	end
	
	gradient(loss, W, b, x)
end

# ╔═╡ eed64d50-34bc-11eb-129b-81211b1951c2
begin
	using Flux.Optimise: train!
	
	train!(loss, params(model), data, optimiser)
end

# ╔═╡ 6c7ee8f0-34ba-11eb-25a0-873babf85e06
md"
## Abstraction Levels
"

# ╔═╡ 768a2b20-34ba-11eb-0207-3b220ac451ef
md"
### Working with Parameters and Gradients
"

# ╔═╡ 005d69b0-34bc-11eb-3b66-bd9c5108700c
md"
Involves computing gradients and applying updates manually (e.g. in a custom training loop).
"

# ╔═╡ e984b810-34bb-11eb-3716-eb8965e541db
md"
### Built-in Optimisers
"

# ╔═╡ 1391cb70-34bc-11eb-30df-d15415209a1a
md"
Using Flux's `update!` function and built-in optimisers to handle parameters update.
"

# ╔═╡ ce4c8770-34bc-11eb-1640-9f4c9900de22
md"
### Standard Training Loop
"

# ╔═╡ 2ebe9f80-34bd-11eb-27bb-3f246ac203be
md"
Using the built-in `train!` function for standard training loop.
"

# ╔═╡ 852c3380-34ba-11eb-2932-b91278d00f9f
begin
	η = 0.1
	θ = params(model)
	for batch in training_set
		grad = gradient(θ) do
			l = loss(batch)
		end
		for p in θ
			p .-= η .* grad[p]
		end
	end
end

# ╔═╡ ce52e042-3320-11eb-0679-b9f3bab9f599
begin
	# Initialise parameters
	W = randn(3, 5)
	b = zeros(3)
	
	function predict(x)
		return sum(W * x .+ b)
	end
	
	function loss(x, y)
		ŷ = predict(x)
		return sum((y .- ŷ).^2)
	end
	
	# Dummy data
	x = rand(5)
	y = rand(2)
	
	θ = params(W, b)
	# gs = gradient(() -> loss(x, y), θ)
	grad = gradient(θ) do
		loss(x, y)
	end
	grad[W], grad[b]
end

# ╔═╡ fb7723ee-34bb-11eb-197b-3938a0e72c36
begin
	using Flux.Optimise: update!
	using Flux.Optimise: Descent
	
	optimiser = Descent(η)
	
	θ = params(model)
	grad = gradient(θ) do
		loss(data)
	end
	
	update!(optimiser, θ, grad)
end

# ╔═╡ Cell order:
# ╟─4fb21b40-331e-11eb-3dd2-fdd16a08c245
# ╠═a77091e0-331e-11eb-09fc-f947365a92b6
# ╟─e8829700-331e-11eb-0f29-2dcab2f31f9a
# ╟─ef8a4a30-33ef-11eb-1717-338b95c94bce
# ╠═f08278d0-331e-11eb-280d-cf00798786a3
# ╠═d679c780-331f-11eb-0421-d56b7f761d15
# ╠═b08808c0-331f-11eb-1901-a1ce85ac5400
# ╠═e64b4f30-331f-11eb-1282-518125edafd7
# ╟─2a2e4ea0-3320-11eb-1091-733f1268d24f
# ╠═40dd3120-3320-11eb-158a-ef5f9c7a54d2
# ╠═ce52e042-3320-11eb-0679-b9f3bab9f599
# ╟─6c7ee8f0-34ba-11eb-25a0-873babf85e06
# ╟─768a2b20-34ba-11eb-0207-3b220ac451ef
# ╟─005d69b0-34bc-11eb-3b66-bd9c5108700c
# ╠═852c3380-34ba-11eb-2932-b91278d00f9f
# ╟─e984b810-34bb-11eb-3716-eb8965e541db
# ╟─1391cb70-34bc-11eb-30df-d15415209a1a
# ╠═fb7723ee-34bb-11eb-197b-3938a0e72c36
# ╟─ce4c8770-34bc-11eb-1640-9f4c9900de22
# ╟─2ebe9f80-34bd-11eb-27bb-3f246ac203be
# ╠═eed64d50-34bc-11eb-129b-81211b1951c2
