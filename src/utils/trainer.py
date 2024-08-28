import math
from pathlib import Path
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from accelerate import Accelerator
import torch
from datetime import datetime
import os
from aim import Run  # Add this import
import nibabel as nib
import numpy as np
from ema_pytorch import EMA
from torch.optim.lr_scheduler import ReduceLROnPlateau

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        model,
        train_dataloader,
        valid_dataloader,
        gradient_accumulate_every = 5,
        train_num_steps = 100000,
        valid_every = 1000,
        save_every = 5000,
        ema_update_every = 10,
        ema_decay = 0.995,
        use_ema = True,
        results_folder = 'results',
        amp = False,
        fp16 = False,
        split_batches = True,
        resume_from = None,
        name = None
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = model

        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.valid_every = valid_every
        self.save_every = save_every

        self.train_dl = self.accelerator.prepare(train_dataloader)
        self.valid_dl = self.accelerator.prepare(valid_dataloader)
        self.train_dl = cycle(self.train_dl)

        # Create a unique folder based on date and time
        if name is not None:
            model.set_name(name)
            print('Model name set to:', name)

        try:
            model_name = self.model.get_name()
        except:
            print('No model name available. Using current date and time.')
            model_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        base_results_folder = Path(results_folder)
        
        self.results_folder = Path(os.path.join(base_results_folder, model_name))
        self.results_folder.mkdir(parents=True, exist_ok=True)

        print(f"Results will be saved in: {self.results_folder}")

        # step counter state
        self.step = 0

        self.use_ema = use_ema
        if self.use_ema and self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.accelerator.device)
        else:
            self.ema = None

        # Add scheduler
        self.scheduler = ReduceLROnPlateau(self.model.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # prepare model, optimizer with accelerator
        self.model, self.model.optimizer = self.accelerator.prepare(self.model, self.model.optimizer)

        self.best_val_loss = float('inf')

        # Initialize or load AIM logger
        if resume_from:
            print(f"Resuming training from checkpoint: {resume_from}")
            self.load(resume_from)
        else:
            print("Starting new training session")
            self.aim_run = Run(repo=self.results_folder, experiment=model_name)
            self.aim_run["hparams"] = {
                "model": model_name,
                "gradient_accumulate_every": gradient_accumulate_every,
                "train_num_steps": train_num_steps,
                "valid_every": valid_every,
                "save_every": save_every,
                "amp": amp,
                "fp16": fp16,
                "split_batches": split_batches,
                "use_ema": use_ema
            }

    def save(self, val_loss):
        if not self.accelerator.is_local_main_process:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            model_state = self.accelerator.get_state_dict(self.model)
            
            # Save model weights
            torch.save(model_state, str(self.results_folder / f'model-best-weights.pt'))
            
            # Save EMA weights if used
            if self.use_ema:
                ema_state = self.ema.state_dict()
                torch.save(ema_state, str(self.results_folder / f'model-best-ema-weights.pt'))
            
            # Save other data
            other_data = {
                'step': self.step,
                'opt': self.model.optimizer.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
                'best_val_loss': self.best_val_loss,
                'scheduler': self.scheduler.state_dict(),
                'use_ema': self.use_ema
            }
            torch.save(other_data, str(self.results_folder / f'model-best-other.pt'))

            # Save AIM run hash
            with open(str(self.results_folder / 'aim_run_hash.txt'), 'w') as f:
                f.write(self.aim_run.hash)

            print(f"\nNew best model saved at step {self.step} with validation loss: {val_loss:.4f}")

    def load(self, checkpoint_folder):
        accelerator = self.accelerator
        device = accelerator.device

        # Load model weights
        model_path = os.path.join(checkpoint_folder, 'model-best-weights.pt')
        model_state = torch.load(model_path, map_location=device)
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(model_state)

        try:
            # Load other data
            other_data_path = os.path.join(checkpoint_folder, 'model-best-other.pt')
            other_data = torch.load(other_data_path, map_location=device)

            # Load optimizer state
            self.model.optimizer.load_state_dict(other_data['opt'])

            self.step = other_data.get('step', 0)
            self.best_val_loss = other_data.get('best_val_loss', float('inf'))

            if exists(self.accelerator.scaler) and 'scaler' in other_data:
                self.accelerator.scaler.load_state_dict(other_data['scaler'])

            # Load scheduler state
            if 'scheduler' in other_data:
                self.scheduler.load_state_dict(other_data['scheduler'])

            # Load EMA weights if used
            self.use_ema = other_data.get('use_ema', False)
            if self.use_ema:
                ema_path = os.path.join(checkpoint_folder, 'model-best-ema-weights.pt')
                if os.path.exists(ema_path):
                    ema_state = torch.load(ema_path, map_location=device)
                    self.ema.load_state_dict(ema_state)
                else:
                    print("EMA weights not found. Initializing EMA with current model weights.")
                    self.ema = EMA(self.model, beta=self.ema.beta, update_every=self.ema.update_every)
            else:
                self.ema = None
        except Exception as e:
            print(f"Warning: Failed to load other data. Error: {str(e)}")
            print("Continuing with default values for step and best_val_loss.")
            self.step = 0
            self.best_val_loss = float('inf')

        try:
            # Load AIM run
            aim_run_hash_file = os.path.join(checkpoint_folder, 'aim_run_hash.txt')
            if os.path.exists(aim_run_hash_file):
                with open(aim_run_hash_file, 'r') as f:
                    run_hash = f.read().strip()
                self.aim_run = Run(repo=self.results_folder, run_hash=run_hash)
                print(f"Resumed AIM run with hash: {run_hash}")
            else:
                raise FileNotFoundError("AIM run hash file not found.")
        except Exception as e:
            print(f"Warning: Failed to load AIM run. Error: {str(e)}")
            print("Starting a new AIM run.")
            self.aim_run = Run(experiment=self.model.get_name())

        print(f"Successfully resumed from checkpoint folder: {checkpoint_folder}")
        print(f"Current step: {self.step}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                loss_dict = {}

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.train_dl)

                    with self.accelerator.autocast():
                        model_output = self.model(batch)
                        
                        if isinstance(model_output, tuple):
                            loss, batch_loss_dict = model_output
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()
                            
                            for key, value in batch_loss_dict.items():
                                if key not in loss_dict:
                                    loss_dict[key] = 0
                                loss_dict[key] += value / self.gradient_accumulate_every
                        else:
                            loss = model_output / self.gradient_accumulate_every
                            total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Track losses
                self.aim_run.track(total_loss, name="train_loss", step=self.step, context={'subset': 'train'})
                for key, value in loss_dict.items():
                    self.aim_run.track(value, name=f"train_{key}_loss", step=self.step, context={'subset': 'train'})
                
                # Update progress bar description
                if loss_dict:
                    pbar_desc = f'loss: {total_loss:.4f} ' + ' '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
                else:
                    pbar_desc = f'loss: {total_loss:.4f}'
                pbar.set_description(pbar_desc)

                accelerator.wait_for_everyone()

                self.model.optimizer.step()
                self.model.optimizer.zero_grad()

                # Update EMA model if used
                if self.use_ema:
                    self.ema.update()

                accelerator.wait_for_everyone()

                self.step += 1

                if self.step != 0 and self.step % self.valid_every == 0:
                    val_loss = self.validate()
                    self.aim_run.track(val_loss, name="val_loss", step=self.step, context={'subset': 'val'})
                    
                    # Step the scheduler
                    self.scheduler.step(val_loss)
                    
                    if self.step % self.save_every == 0:
                        self.save(val_loss)

                pbar.update(1)

        accelerator.print('training complete')

    def validate(self):
        accelerator = self.accelerator
        device = accelerator.device

        self.model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch in self.valid_dl:
                output, label = self.model.valid_step(batch)
                if isinstance(self.model.criterion, str):
                    loss = torch.nn.functional.mse_loss(output, label)
                else:
                    loss = self.model.criterion(output, label)

                val_loss += loss.item()

        val_loss /= len(self.valid_dl)
        accelerator.print(f'Validation Loss: {val_loss:.4f}')
        self.model.train()
        return val_loss

    def __del__(self):
        # Close the AIM run when the Trainer object is destroyed
        if hasattr(self, 'aim_run'):
            self.aim_run.close()


class Inferer:
    def __init__(
        self,
        model,
        test_dataloader,
        results_folder='results',
        fp16=False,
        split_batches=True,
        checkpoint_path=None
    ):
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )

        self.model = model
        self.test_dl = self.accelerator.prepare(test_dataloader)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # Prepare model with accelerator
        self.model = self.accelerator.prepare(self.model)

        if checkpoint_path:
            self.load(checkpoint_path)

    def load(self, checkpoint_path):
        device = self.accelerator.device
        model_state = torch.load(checkpoint_path, map_location=device)
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(model_state)
        print(f"Model loaded from: {checkpoint_path}")

    def infer(self):
        self.model.eval()
        all_inputs = []
        all_outputs = []
        all_labels = []

        with torch.inference_mode():
            for batch in tqdm(self.test_dl, desc="Inferring", disable=not self.accelerator.is_main_process):
                outputs, labels = self.model.infer_step(batch)
                all_inputs.append(batch['image'].cpu())
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        # Gather all outputs and labels across processes
        all_inputs = self.accelerator.gather(torch.cat(all_inputs))
        all_outputs = self.accelerator.gather(torch.cat(all_outputs))
        all_labels = self.accelerator.gather(torch.cat(all_labels))

        return all_inputs, all_outputs, all_labels

    def save_results(self, inputs, outputs, labels, filename_prefix='results'):
        if self.accelerator.is_main_process:
            # Ensure outputs and labels are numpy arrays
            inputs_np = inputs.numpy().squeeze()
            outputs_np = outputs.numpy().squeeze()
            labels_np = labels.numpy().squeeze()
            
            inputs_np = np.moveaxis(inputs_np, 0, -1)
            outputs_np = np.moveaxis(outputs_np, 0, -1)
            labels_np = np.moveaxis(labels_np, 0, -1)

            # Save inputs as NIfTI
            inputs_nifti = nib.Nifti1Image(inputs_np, np.eye(4))
            nib.save(inputs_nifti, self.results_folder / f'{filename_prefix}_inputs.nii.gz')
            
            # Save outputs as NIfTI
            outputs_nifti = nib.Nifti1Image(outputs_np, np.eye(4))
            nib.save(outputs_nifti, self.results_folder / f'{filename_prefix}_outputs.nii.gz')

            # Save labels as NIfTI
            labels_nifti = nib.Nifti1Image(labels_np, np.eye(4))
            nib.save(labels_nifti, self.results_folder / f'{filename_prefix}_labels.nii.gz')

            print(f"Results saved to: {self.results_folder}")
            print(f"Inputs: {self.results_folder / f'{filename_prefix}_inputs.nii.gz'}")
            print(f"Outputs: {self.results_folder / f'{filename_prefix}_outputs.nii.gz'}")
            print(f"Labels: {self.results_folder / f'{filename_prefix}_labels.nii.gz'}")

    def run(self):
        inputs, outputs, labels = self.infer()
        self.save_results(inputs, outputs, labels)
        return outputs, labels