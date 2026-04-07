import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out

class DiffusionTrainer(nn.Module):
    def __init__(self, model: nn.Module, beta = [0.0001, 0.02], T = 1000):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # calculate and store two coefficient of $q(x_t | x_0)$
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))

    def forward(self, x_0, mri, cate_x, conti_x):
        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0)

        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
               extract(self.noise_rate, t, x_0.shape) * epsilon)
        epsilon_theta = self.model(x_t, mri, cate_x, conti_x, t)

        # get the gradient
        loss = F.mse_loss(epsilon_theta, epsilon)
        return loss

class DDIMSampler(nn.Module):
    def __init__(self, model, beta = [0.0001, 0.02], T = 1000):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    def sample_one_step(self, x_t,mri,cate_x,conti_x, time_step: int, prev_time_step: int, eta: float):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict noise using model
        epsilon_theta_t = self.model(x_t,mri,cate_x,conti_x, t)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )
        return x_t_minus_one

    def forward(self, x_t,mri,cate_x,conti_x, steps: int = 100, method="linear", eta=0.0,
                only_return_x_0: bool = True, interval: int = 50):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            steps: Sampling steps.
            method: Sampling method, can be "linear" or "quadratic".
            eta: Coefficients of sigma parameters in the paper. The value 0 indicates DDIM, 1 indicates DDPM.
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures, according to `step`.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int)
        else:
            raise NotImplementedError(f"sampling method {method} is not implemented!")

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        time_steps = time_steps + 1
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x = [x_t]
        with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps) as sampling_steps:
            for i in sampling_steps:
                x_t = self.sample_one_step(x_t,mri,cate_x,conti_x, time_steps[i], time_steps_prev[i], eta)

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": i + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]