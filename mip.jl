using JuMP
using GLPK
# using Gurobi
using Plots
using LinearAlgebra
using Random

# Number of points and dimension
n = 80
d = 2

R = .3
Random.seed!(42)

# Point locations
# Hacky, points should be outside the obstacle, so no norm greater than 1
#P = randn(d,n)
P = (rand(d,n).-.5)*10
ix = findall(vec(sum(P.^2,dims=1)) .> 4*R^2);
P = P[:, ix]
n = size(P,2)

# Closest distance to the origin in line segment x to y
min_dist(x,y) = minimum([norm(t*x+(1-t)*y) for t in range(0,1,400)])

# This is a hack, only obstacle is the unit disk
function visibility_graph(P,R)
    d,n = size(P)
    G = zeros(Int,n,n)
    for j=1:n
        for k = (j+1):n
            if min_dist(P[:,j],P[:,k])>R 
                G[j,k]=G[k,j]=1
            end
        end
    end
    return G
end


# Visibility graph (hack for now)
G = visibility_graph(P,R)

# Create a model using Gurobi
model = Model(GLPK.Optimizer)
#unset_silent(model)
#set_optimizer_attribute(model, "msg_lev", GLPK.GLP_MSG_ALL)

# Define decision variables
@variable(model, x[1:n], Bin) 
@variable(model, q[1:d,1:n])

# Define the objective function (minimize total cost)
obj = sum(x)
@objective(model, Max, obj)

# Clique constraint
for j=1:n
    for k = (j+1):n
        if G[j,k]==0 @constraint(model,x[j]+x[k]<=1) end
    end
end

# Spatial constraint
for k=1:n
    for j=1:n
        if j != k
        @constraint(model, (P[:,j]-P[:,k])'*q[:,k] >= x[j]-x[k]-1+1e-4)
        end
    end
end
# for k=1:d
#     for j=1:n
#         @constraint(model, q[k,j]==0)
#     end
# end


# Solve the problem
optimize!(model)

# Print the results
println("Objective Value: ", value(obj))
println("Solution:")
println( value.(x) )


########

# Aux functions for plotting
# Hacky stuff, there should be a nicer way
# scatter works, but can't control the markersize nicely to match scale

function circle(x, y, r=1; n=64)
    t = range(0,2pi,length=n)
    Plots.Shape(x .+ r*sin.(t), y .+ r*cos.(t))
end

#######################


nc = 1
#C = [ [-6 ; 0]   [6 ; 1] ]
#R = [ 1 ; 2]

CC = [ [ 0 ; 0] ; ]
RR = [ R ; ]
RR = [ R ; ]
# Plot circles
plot_args = (ratio=:equal, legend=false, xlim=(-5.5,5.5), ylim = (-5.5,5.5) )


h=plot([circle( CC[1,j], CC[2,j], RR[j] ) for j in 1:nc]; plot_args...)


# Plot nodes and edges
for j = 1:n
    # Edges
    for k =(j+1):n
        if G[j,k]==1
        p1 = P[:,j]
        p2 = P[:,k]
        plot!( [p1[1] , p2[1]], [p1[2] , p2[2]], color=:pink) 
        end
    end 
end

# Nodes
for j = 1:n
    if value(x[j])==1;
        plot!(circle( P[1,j], P[2,j], .12 ) ; plot_args...)
    end
end
for j = 1:n
    plot!(circle( P[1,j], P[2,j], .05 ), color=:black ; plot_args...)
end

display(h)
