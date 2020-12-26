### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ 4f92ba52-46d0-11eb-1bf9-db7b46c09700
begin
	using Plots
	using Flux
end

# ╔═╡ 36471440-46cd-11eb-085a-afcbdf57fe08
md"
# Logarithm AD
"

# ╔═╡ d5ea2e40-4742-11eb-1079-7b0dc7c15a69
md"
## Model
"

# ╔═╡ 79906bc0-46cd-11eb-37f3-bd8287b09a0d
md"
Let

$f(x, y) = \frac{((4*\text{round}(0.7x)) + 30x + 56) + y(\text{round}(0.7x) + 2x + 2)}{1 + \frac{x}{10}}$

Our goal is to find a simpler form of this formula.
"

# ╔═╡ adafd0c0-46ce-11eb-137d-dd449d79d6ae
md"
First, we split the formula into the following terms:

$term_{base} = \frac{(4*\text{round}(0.7x)) + 30x + 56}{1 + \frac{x}{10}}$

$term_{multiplier} = y\frac{\text{round}(0.7x) + 2x + 2}{1 + \frac{x}{10}}$
"

# ╔═╡ bf748070-46cf-11eb-0437-338df628df4d
md"
Then we note that the base term and the multiplier coefficient grows logarithimcally with $x$.
"

# ╔═╡ 9c8536d0-46d0-11eb-0998-e137948d4976
begin
	step(x) = round.(0.7 .* x)
	denominator(x) = 1 .+ (x ./ 10)
	
	base_term(x) = ((4 .* step.(x)) .+ (30 .* x) .+ 56) / denominator.(x)
	y_coefficient(x) = (step.(x) + (2 .* x) .+ 2) / denominator.(x)
end

# ╔═╡ 1dac4370-46d1-11eb-05bf-8bfad32cb210
plot(base_term, 0, 50, xaxis="x", yaxis="f", label="", title="Base term")

# ╔═╡ 62c7f480-46d2-11eb-0c67-692e38bfdd81
plot(y_coefficient, 0, 50, xaxis="x", yaxis="f", label="", title="y-coefficient")

# ╔═╡ 88ce259e-46d2-11eb-3d6c-37231f2b6f68
md"
Assuming the data source that the formula was derived from was rounded to integral values, it is natural that the plot will contain irregular steps and that the true formula will most likely be the logarithm that these plots are approximating.
The logarithm will also likely lead to further simplications down the line.
"

# ╔═╡ e5d0d1c0-46d8-11eb-351b-4dd5cf6c3e16
md"
That said, we should also expect an exact match (up to rounding to the nearest integral values) from the resulting model.
Otherwise, the model is most likely incorrect and we should consider other options.
"

# ╔═╡ de0b7ac0-4742-11eb-3599-4553c2d28126
md"
## AD
"

# ╔═╡ 68346fe0-4743-11eb-0df9-7787af5c654e
md"
$model(x) = m_{1} \text{log}_{b}{(m_{2}x)} + c$
"

# ╔═╡ e7df0b20-4742-11eb-2573-27d4c8aaf6cc
begin
	m1 = 1
	m2 = 1
	b = 2
	c = 0
	model(x) = (m1 .* log.(b, m2 .* x)) .+ c
	loss(x, y) = Flux.Losses.mse(model.(x), y)
	params = Flux.params([m1, m2, b, c])
end

# ╔═╡ 71a7fe92-4746-11eb-018d-1f7b3d3ad1ba
begin
	x = rand(1:200, 1000)
	y = base_term.(x)
	data = Flux.Data.DataLoader(x, y, batchsize=4)
	optimiser = Flux.Optimise.ADAM(0.001)
end

# ╔═╡ e9361600-474a-11eb-28d2-196e990a525b
callback() = @show(loss(1:10, base_term.(1:10)))

# ╔═╡ 35188970-4748-11eb-1ad7-e54d2eecfc2f
Flux.train!(loss, params, data, optimiser, cb=callback)

# ╔═╡ Cell order:
# ╟─36471440-46cd-11eb-085a-afcbdf57fe08
# ╠═4f92ba52-46d0-11eb-1bf9-db7b46c09700
# ╟─d5ea2e40-4742-11eb-1079-7b0dc7c15a69
# ╟─79906bc0-46cd-11eb-37f3-bd8287b09a0d
# ╟─adafd0c0-46ce-11eb-137d-dd449d79d6ae
# ╟─bf748070-46cf-11eb-0437-338df628df4d
# ╠═9c8536d0-46d0-11eb-0998-e137948d4976
# ╟─1dac4370-46d1-11eb-05bf-8bfad32cb210
# ╟─62c7f480-46d2-11eb-0c67-692e38bfdd81
# ╟─88ce259e-46d2-11eb-3d6c-37231f2b6f68
# ╟─e5d0d1c0-46d8-11eb-351b-4dd5cf6c3e16
# ╟─de0b7ac0-4742-11eb-3599-4553c2d28126
# ╟─68346fe0-4743-11eb-0df9-7787af5c654e
# ╠═e7df0b20-4742-11eb-2573-27d4c8aaf6cc
# ╠═71a7fe92-4746-11eb-018d-1f7b3d3ad1ba
# ╠═e9361600-474a-11eb-28d2-196e990a525b
# ╠═35188970-4748-11eb-1ad7-e54d2eecfc2f
