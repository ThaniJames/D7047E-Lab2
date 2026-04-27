# D7047E — Lab 2: GAN & Diffusion

PyTorch implementations for *Practical Exercise 2* on MNIST.

## Layout

| File | Task |
|---|---|
| [Task_1_2.ipynb](Task_1_2.ipynb) | Vanilla GAN with BCE loss and with logistic loss |
| [cGAN.ipynb](cGAN.ipynb) | Conditional GAN |
| [Task4_Adversarial.ipynb](Task4_Adversarial.ipynb) | CNN classifier + adversarial attacks |
| [Diffusion.ipynb](Diffusion.ipynb) | Conditional DDPM |
| [GAN.py](GAN.py) | Reference vanilla GAN |

## Tasks

- **Task 1 — Vanilla GAN (BCE).** Generator and Discriminator are simple MLPs trained adversarially with binary cross-entropy.
- **Task 2 — Logistic loss.** Same architecture as Task 1 with the Discriminator's sigmoid removed and the loss rewritten with softplus on the logits. Both losses are run side by side at three epoch budgets and compared with sample grids and loss curves.
- **Task 3 — Conditional GAN.** Adds a learned label embedding that is concatenated with the noise vector for the Generator and with the flattened image for the Discriminator. A separately-trained CNN evaluator scores the Generator each epoch and is used to checkpoint the best one.
- **Task 4 — Adversarial images.** A small CNN is trained on MNIST and then attacked: a targeted iterative sign attack flips a real `4` into a `9` with an almost invisible perturbation, and a gradient-descent attack starting from random noise produces an image the model is certain is a `9` but a human cannot read.
- **Task 5 — Stable diffusion.** A conditional U-Net is trained, in the DDPM formulation, to predict the noise added to a clean digit at a random timestep. Generation walks the noise schedule in reverse to turn pure noise into a chosen digit, and a short discussion compares the result to the conditional GAN of Task 3.