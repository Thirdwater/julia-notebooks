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
	gs = gradient(θ) do
		loss(x, y)
	end
	gs[W], gs[b]
end

# ╔═╡ Cell order:
# ╟─4fb21b40-331e-11eb-3dd2-fdd16a08c245
# ╠═a77091e0-331e-11eb-09fc-f947365a92b6
# ╟─e8829700-331e-11eb-0f29-2dcab2f31f9a
# ╠═f08278d0-331e-11eb-280d-cf00798786a3
# ╠═d679c780-331f-11eb-0421-d56b7f761d15
# ╠═b08808c0-331f-11eb-1901-a1ce85ac5400
# ╠═e64b4f30-331f-11eb-1282-518125edafd7
# ╟─2a2e4ea0-3320-11eb-1091-733f1268d24f
# ╠═40dd3120-3320-11eb-158a-ef5f9c7a54d2
# ╠═ce52e042-3320-11eb-0679-b9f3bab9f599
