### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 167caec0-06fb-11eb-02b0-51ff7f64a2e9
begin
	using Flux
end

# ╔═╡ 7ca33760-070e-11eb-2855-85078f28ac32
md"
# Reinforcement Learning
"

# ╔═╡ b0231620-0711-11eb-2cd5-23af8b0f6794
md"
## General Framework
"

# ╔═╡ 9d35138e-070e-11eb-08ce-f55394760595
begin
	
	struct Environment{StateSpace, ActionSpace}
		state::StateSpace
	end
	
	function reset(env::Environment{S, A})::S where {S, A}
		return zeros(StateSpace)
	end
	
	function step(env::Environment{S, A}, action::A) where {S, A}
		return zeros(StateSpace)
	end
	
end

# ╔═╡ f0471920-070e-11eb-023e-999fb917ba55
struct Observation{}
	
end

# ╔═╡ bc003220-0711-11eb-2076-a70ef9a437e7
md"
## Tic-Tac-Toe Example
"

# ╔═╡ Cell order:
# ╠═167caec0-06fb-11eb-02b0-51ff7f64a2e9
# ╟─7ca33760-070e-11eb-2855-85078f28ac32
# ╟─b0231620-0711-11eb-2cd5-23af8b0f6794
# ╠═9d35138e-070e-11eb-08ce-f55394760595
# ╠═f0471920-070e-11eb-023e-999fb917ba55
# ╟─bc003220-0711-11eb-2076-a70ef9a437e7
