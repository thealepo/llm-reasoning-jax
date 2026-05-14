from model import PolicyModel
import flax.nnx as nnx


m = PolicyModel(4 , 32 , 1 , rngs=nnx.Rngs(42))

print(m)