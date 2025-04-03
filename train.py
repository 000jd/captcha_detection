from src.train_test import train_model, prepare_training_data

# Prepare data
yaml_path = prepare_training_data(
    input_dir='data/raw',
    output_dir='data/processed',
    augment=True,
    num_augmentations = 5
)

# Train model
model_path = train_model(
    data_yaml=yaml_path,
    model_size='n',
    epochs=200,
    batch_size=32,
    img_size = 640,
    save_dir = 'runs/train',
)