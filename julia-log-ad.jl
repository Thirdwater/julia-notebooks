### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ 4f92ba52-46d0-11eb-1bf9-db7b46c09700
begin
	using Plots
end

# ╔═╡ 36471440-46cd-11eb-085a-afcbdf57fe08
md"
# Logarithmic AD
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
	step(x) = round(0.7*x)
	denominator(x) = 1 + (x/10)
	
	base_term(x) = ((4*step(x)) + 30*x + 56) / denominator(x)
	y_coefficient(x) = (step(x) + 2*x + 2) / denominator(x)
end

# ╔═╡ 1dac4370-46d1-11eb-05bf-8bfad32cb210
plot(base_term, 0, 50, xaxis="x", yaxis="f", label="", title="Base term")

# ╔═╡ 62c7f480-46d2-11eb-0c67-692e38bfdd81
plot(y_coefficient, 0, 50, xaxis="x", yaxis="f", label="", title="y-coefficient")

# ╔═╡ 88ce259e-46d2-11eb-3d6c-37231f2b6f68
md"
Assuming the data source that the formula was derived from was rounded to integral values, it is natural that the plot will contain irregular steps and that the true formula will most likely be the logarithm that these plots are approximating.
The logarithmic form will also likely lead to further simplications down the line.
"

# ╔═╡ e5d0d1c0-46d8-11eb-351b-4dd5cf6c3e16
md"
That said, we should also expect an exact match (up to rounding to the nearest integral values) from the resulting model.
Otherwise, we should consider a different model.
"

# ╔═╡ Cell order:
# ╟─36471440-46cd-11eb-085a-afcbdf57fe08
# ╠═4f92ba52-46d0-11eb-1bf9-db7b46c09700
# ╟─79906bc0-46cd-11eb-37f3-bd8287b09a0d
# ╟─adafd0c0-46ce-11eb-137d-dd449d79d6ae
# ╟─bf748070-46cf-11eb-0437-338df628df4d
# ╠═9c8536d0-46d0-11eb-0998-e137948d4976
# ╟─1dac4370-46d1-11eb-05bf-8bfad32cb210
# ╟─62c7f480-46d2-11eb-0c67-692e38bfdd81
# ╟─88ce259e-46d2-11eb-3d6c-37231f2b6f68
# ╟─e5d0d1c0-46d8-11eb-351b-4dd5cf6c3e16
