import autorootcwd
import src.data
import src.losses
import src.metrics
import src.models

import click

from src.models.supervised_model import SupervisedModel
from src.utils.registry import DATASET_REGISTRY, ARCH_REGISTRY, LOSS_REGISTRY
from torch.utils.data import DataLoader
from src.utils.trainer import Trainer

@click.command()
@click.option('--dset', type=click.Choice(['OCTA500_6M', 'OCTA500_3M', 'ROSSA']), default='ROSSA', help='Dataset to use')
@click.option('--network', type=click.Choice(list(ARCH_REGISTRY.keys())), default='AttentionUnet', help='Network architecture')
@click.option('--loss', type=click.Choice(list(LOSS_REGISTRY.keys())), default='MonaiDiceCELoss', help='Loss function')
@click.option('--train-batch-size', default=16, help='Training batch size')
@click.option('--valid-batch-size', default=8, help='Validation batch size')
@click.option('--train-num-steps', default=5000, help='Number of training steps')
@click.option('--valid-every', default=100, help='Validate every N steps')
@click.option('--save-every', default=500, help='Save model every N steps')
@click.option('--interactive', is_flag=True, help='Run in interactive mode')
def main(dset, network, loss, train_batch_size, valid_batch_size, train_num_steps, valid_every, save_every, interactive):
    if interactive:
        dset = click.prompt('Dataset Name', type=click.Choice(['OCTA500_6M', 'OCTA500_3M', 'ROSSA']), default=dset)
        network = click.prompt('Network', type=click.Choice(list(ARCH_REGISTRY.keys())), default=network)
        loss = click.prompt('Loss function', type=click.Choice(list(LOSS_REGISTRY.keys())), default=loss)
        train_batch_size = click.prompt('Training batch size', type=int, default=train_batch_size)
        valid_batch_size = click.prompt('Validation batch size', type=int, default=valid_batch_size)
        train_num_steps = click.prompt('Number of training steps', type=int, default=train_num_steps)
        valid_every = click.prompt('Validate every N steps', type=int, default=valid_every)
        save_every = click.prompt('Save model every N steps', type=int, default=save_every)

    path_dict = {
        dset: f"data/{dset}"
    }

    datasets_collapsed = DATASET_REGISTRY.get('VesselDataset')(path_dict, collapse=True)
    print("Collapsed datasets:")
    for split, dataset in datasets_collapsed.items():
        print(f"{split}: {len(dataset)} samples")

    train_dataset = datasets_collapsed['train']
    valid_dataset = datasets_collapsed['val']
    test_dataset = datasets_collapsed['test']

    print('Length of datasets:')
    print(len(train_dataset), len(valid_dataset), len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=4, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, num_workers=4, shuffle=False)

    model = SupervisedModel(arch=network, criterion=loss, mode='train')
    model_folder_name = model.get_name()
    dataset_model_folder_name = model_folder_name + '_' + dset
    
    model.set_name(dataset_model_folder_name)
    print('Saving the result to the folder:', dataset_model_folder_name)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        train_num_steps=train_num_steps,
        valid_every=valid_every,
        save_every=save_every
    )
    trainer.train()

if __name__ == '__main__':
    main()