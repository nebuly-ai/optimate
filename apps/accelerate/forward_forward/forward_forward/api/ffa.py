from forward_forward import train_with_forward_forward_algorithm
import os 
os.environ["CUDA"]="-1"

trained_model = train_with_forward_forward_algorithm(
    model_type="progressive",
    n_layers=3,
    hidden_size=2000,
    lr=0.03,
    device="cpu",
    epochs=100,
    batch_size=5000,
    theta=2.,
)