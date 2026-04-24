# Call main class from model.py file
from models.model import MultimodalClassifier

def print_model_parameters(model_name, model_kwargs):
    # Create model
    model = MultimodalClassifier(**model_kwargs)
    
    # Calculate total parameter numbers
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{model_name} Model] Total trainable parameters: {total_params:,}")
    
    # Print the number of parameters for each submodule.
    print("-" * 50)
    for name, child in model.named_children():
        child_params = sum(p.numel() for p in child.parameters() if p.requires_grad)
        print(f"{name:30} : {child_params:,}")
    print("=" * 50)

def main():
    print("Calculating parameter counts for all model architectures...\n")
    
    # 1. Baseline Model (2 layers, 128 d_model)
    baseline_kwargs = {
        'd_model': 128, 
        'nhead': 4, 
        'num_layers': 2, 
        'dim_feedforward': 512
    }
    print_model_parameters("Baseline", baseline_kwargs)
    
    # 2. Large Model (3 layers, 256 d_model, 1024 ff_dim)
    large_kwargs = {
        'd_model': 256, 
        'nhead': 8, 
        'num_layers': 3, 
        'dim_feedforward': 1024
    }
    print_model_parameters("Large", large_kwargs)
    
    # 3. Deep Model (4 layers, 128 d_model, 0.2 dropout)
    deep_kwargs = {
        'd_model': 128, 
        'nhead': 4, 
        'num_layers': 4, 
        'dim_feedforward': 512,
        'dropout': 0.2
    }
    print_model_parameters("Deep", deep_kwargs)

if __name__ == "__main__":
    main()