import numpy as np


class DomainRouter:
    def __init__(self, model, lora_names, apply_single):
        self.model = model
        self.lora_names = list(lora_names)
        self.apply_single = apply_single
        self.domain_nlls = {}

    def fit(self, adapt_prompts, verbose=False):
        if not self.lora_names:
            raise ValueError("At least one adapter is required")
        losses = []
        for domain in self.lora_names:
            self.apply_single(domain)
            loss = float(self.model.domain_nll(adapt_prompts))
            losses.append(loss)
            self.domain_nlls[domain] = loss
            if verbose:
                print(f"Router NLL [{domain}]: {loss:.4f}")
        values = np.asarray(losses, dtype=float)
        logits = -values / max(float(values.std()), 0.01)
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        if verbose:
            print(f"Router weights: {np.round(weights, 3)}")
        return weights
