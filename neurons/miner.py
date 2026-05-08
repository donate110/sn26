import argparse
import logging as pylogging
import os
import time
import typing

import bittensor as bt
import torch
import torch.nn.functional as F

from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import load_efficientnet_b5, logits_for_images, predict_index, resolve_target_index
from perturbnet.protocol import AttackChallenge


def _make_wallet(config):
    wallet_name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    wallet_hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))
    if hasattr(bt, "wallet"):
        try:
            return bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        except Exception:
            return bt.wallet(config=config)
    wallet_cls = getattr(bt, "Wallet", None)
    if wallet_cls is None:
        raise RuntimeError("No wallet constructor found in bittensor.")
    try:
        return wallet_cls(name=wallet_name, hotkey=wallet_hotkey)
    except TypeError:
        return wallet_cls(config=config)


def _make_subtensor(config):
    network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    chain_endpoint = getattr(config.subtensor, "chain_endpoint", None) or getattr(config, "chain_endpoint", None)
    if hasattr(bt, "subtensor"):
        if chain_endpoint:
            try:
                return bt.subtensor(chain_endpoint=chain_endpoint)
            except Exception:
                pass
        try:
            return bt.subtensor(network=network)
        except Exception:
            return bt.subtensor(config=config)
    subtensor_cls = getattr(bt, "Subtensor", None)
    if subtensor_cls is None:
        raise RuntimeError("No subtensor constructor found in bittensor.")
    if chain_endpoint:
        try:
            return subtensor_cls(chain_endpoint=chain_endpoint)
        except Exception:
            pass
    try:
        return subtensor_cls(network=network)
    except Exception:
        return subtensor_cls(config=config)


def _make_axon(wallet, config):
    resolved_config = config() if callable(config) else config
    if hasattr(bt, "axon"):
        try:
            return bt.axon(wallet=wallet, config=resolved_config)
        except Exception:
            return bt.axon(wallet=wallet)
    axon_cls = getattr(bt, "Axon", None)
    if axon_cls is None:
        raise RuntimeError("No axon constructor found in bittensor.")
    try:
        return axon_cls(wallet=wallet, config=resolved_config)
    except Exception:
        return axon_cls(wallet=wallet)


def _configure_log_level(level_raw: str) -> None:
    level_name = (level_raw or "DEBUG").upper()
    level = getattr(pylogging, level_name, pylogging.INFO)
    pylogging.getLogger().setLevel(level)
    bt_logger = getattr(bt, "logging", None)
    if bt_logger is None:
        return
    try:
        if level_name == "DEBUG" and hasattr(bt_logger, "set_debug"):
            bt_logger.set_debug(True)
        elif level_name in {"WARNING", "WARN"} and hasattr(bt_logger, "set_warning"):
            bt_logger.set_warning(True)
        elif level_name == "ERROR" and hasattr(bt_logger, "set_error"):
            bt_logger.set_error(True)
        elif hasattr(bt_logger, "set_info"):
            bt_logger.set_info(True)
    except Exception:
        pass


class PerturbMiner:
    def __init__(self, config: typing.Any) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = self._init_subtensor_with_retry()
        self.metagraph = self._init_metagraph_with_retry()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_efficientnet_b5(self.device)

        self.axon = _make_axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

    def _log_step_start(self, step_name: str, **context: typing.Any) -> None:
        if context:
            rendered = " ".join([f"{k}={v}" for k, v in context.items()])
            bt.logging.info(f"[STEP_START] {step_name} {rendered}")
        else:
            bt.logging.info(f"[STEP_START] {step_name}")

    def _init_subtensor_with_retry(self):
        max_attempts = int(os.getenv("SUBTENSOR_CONNECT_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("SUBTENSOR_CONNECT_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                bt.logging.info(f"[MINER] Connecting subtensor (attempt {attempt}/{max_attempts})")
                return _make_subtensor(config=self.config)
            except Exception as err:
                last_error = err
                bt.logging.warning(f"[MINER] Subtensor connect failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to connect subtensor after {max_attempts} attempts: {last_error}")

    def _init_metagraph_with_retry(self):
        max_attempts = int(os.getenv("METAGRAPH_SYNC_RETRIES", "5"))
        retry_delay_seconds = float(os.getenv("METAGRAPH_SYNC_RETRY_SECONDS", "4"))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                bt.logging.info(f"[MINER] Loading metagraph netuid={self.config.netuid} (attempt {attempt}/{max_attempts})")
                return self.subtensor.metagraph(netuid=self.config.netuid)
            except Exception as err:
                last_error = err
                bt.logging.warning(f"[MINER] Metagraph load failed on attempt {attempt}: {err}")
                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds * attempt)
        raise RuntimeError(f"Failed to load metagraph after {max_attempts} attempts: {last_error}")

    def sync(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)

    def _find_best_target_class(self, clean: torch.Tensor, true_index: int) -> int:
        """Find the second-best class (easiest target to reach)."""
        with torch.no_grad():
            logits = logits_for_images(model=self.model, image_bchw=clean.unsqueeze(0))
            logits = logits[0]  # Remove batch dimension
            # Mask out the true class
            logits[true_index] = float('-inf')
            target_index = int(logits.argmax().item())
        return target_index

    def _find_top_k_targets(self, clean: torch.Tensor, true_index: int, k: int = 5) -> typing.List[int]:
        """Find top-k most likely alternative classes."""
        with torch.no_grad():
            logits = logits_for_images(model=self.model, image_bchw=clean.unsqueeze(0))
            logits = logits[0]  # Remove batch dimension
            # Mask out the true class
            logits[true_index] = float('-inf')
            top_k = torch.topk(logits, k=min(k, logits.size(0)))
            return [int(idx) for idx in top_k.indices.tolist()]
        return []

    def _batch_targeted_mifgsm_attack(
        self,
        clean: torch.Tensor,
        true_index: int,
        target_indices: typing.List[int],
        epsilon: float,
        min_delta: float,
        steps: int = 30,
    ) -> typing.Tuple[torch.Tensor, bool, float]:
        """
        Batched targeted MI-FGSM attack with momentum for multiple targets.
        Leverages 96GB VRAM to process multiple attack paths simultaneously.
        Returns (best_adversarial_image, success_flag, final_norm).
        """
        batch_size = len(target_indices)
        if batch_size == 0:
            return clean, False, 0.0
        
        # Stack clean image for batch processing
        clean_batch = clean.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        # Initialize momentum for each attack path
        momentum_batch = torch.zeros_like(clean_batch)
        decay = 0.9
        step_size = epsilon / 10.0
        
        adv_batch = clean_batch.clone().detach()
        
        # Track best result across all paths
        best_adv = clean.clone()
        best_norm = float('inf')
        found_success = False
        
        target_tensor = torch.tensor(target_indices, device=self.device)
        true_tensor = torch.tensor([true_index] * batch_size, device=self.device)
        
        for step in range(steps):
            adv_batch.requires_grad_(True)
            logits = logits_for_images(model=self.model, image_bchw=adv_batch)
            
            # Targeted loss for each path
            loss = -F.cross_entropy(logits, target_tensor, reduction='sum')
            loss += F.cross_entropy(logits, true_tensor, reduction='sum')
            
            grad = torch.autograd.grad(loss, adv_batch)[0]
            
            # Normalize gradients per image
            for i in range(batch_size):
                grad_norm = grad[i].abs().sum() + 1e-12
                grad[i] = grad[i] / grad_norm
            
            # Update momentum
            momentum_batch = decay * momentum_batch + grad
            
            # Update adversarial images
            adv_batch = adv_batch.detach() + step_size * momentum_batch.sign()
            
            # Project to epsilon ball
            perturbation_batch = adv_batch - clean_batch
            perturbation_batch = torch.clamp(perturbation_batch, -epsilon, epsilon)
            adv_batch = clean_batch + perturbation_batch
            adv_batch = torch.clamp(adv_batch, 0.0, 1.0)
            
            # Check success for each path every few steps
            if step % 5 == 0 or step == steps - 1:
                with torch.no_grad():
                    preds_logits = logits_for_images(model=self.model, image_bchw=adv_batch)
                    preds = preds_logits.argmax(dim=1)
                    
                    for i in range(batch_size):
                        if preds[i].item() != true_index:
                            current_norm = float((adv_batch[i] - clean).abs().max().item())
                            if current_norm >= min_delta and current_norm < best_norm:
                                found_success = True
                                best_adv = adv_batch[i].clone()
                                best_norm = current_norm
        
        return best_adv, found_success, best_norm

    def _adaptive_pgd_attack(
        self,
        clean: torch.Tensor,
        true_index: int,
        target_index: int,
        epsilon: float,
        min_delta: float,
        steps: int = 40,
    ) -> typing.Tuple[torch.Tensor, bool, float]:
        """
        Adaptive PGD with dynamic step size adjustment.
        """
        step_size = epsilon / 4.0
        adv = clean.clone().detach()
        best_adv = adv.clone()
        best_norm = float('inf')
        found_success = False
        
        for step in range(steps):
            adv.requires_grad_(True)
            logits = logits_for_images(model=self.model, image_bchw=adv.unsqueeze(0))
            
            # Targeted loss
            loss = -F.cross_entropy(logits, torch.tensor([target_index], device=self.device))
            loss += 2.0 * F.cross_entropy(logits, torch.tensor([true_index], device=self.device))
            
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + step_size * grad.sign()
            
            # Project
            perturbation = adv - clean
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            adv = clean + perturbation
            adv = torch.clamp(adv, 0.0, 1.0)
            
            # Check and adapt
            pred = predict_index(model=self.model, image_chw=adv)
            current_norm = float((adv - clean).abs().max().item())
            
            if pred != true_index and current_norm >= min_delta:
                found_success = True
                if current_norm < best_norm:
                    best_adv = adv.clone()
                    best_norm = current_norm
                # Reduce step size to refine
                step_size *= 0.7
            elif found_success:
                # Already found success, keep refining
                step_size *= 0.95
            
            if step_size < epsilon / 100:
                break
        
        return best_adv, found_success, best_norm if found_success else float('inf')

    def _cw_style_attack(
        self,
        clean: torch.Tensor,
        true_index: int,
        target_index: int,
        epsilon: float,
        min_delta: float,
        steps: int = 50,
    ) -> typing.Tuple[torch.Tensor, bool, float]:
        """
        Carlini & Wagner style attack with margin loss.
        """
        # Use tanh space for unbounded optimization
        w = torch.zeros_like(clean, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=0.01)
        
        best_adv = clean.clone()
        best_norm = float('inf')
        found_success = False
        
        c = 1.0  # Confidence parameter
        
        for step in range(steps):
            # Map from tanh space to [0, 1]
            adv = (torch.tanh(w) + 1.0) / 2.0
            
            # Project to epsilon ball in original space
            perturbation = adv - clean
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            adv = clean + perturbation
            adv = torch.clamp(adv, 0.0, 1.0)
            
            # Get logits
            logits = logits_for_images(model=self.model, image_bchw=adv.unsqueeze(0))[0]
            
            # Margin loss
            true_logit = logits[true_index]
            target_logit = logits[target_index]
            margin_loss = torch.clamp(true_logit - target_logit + c, min=0.0)
            
            # Perturbation penalty
            perturbation_loss = (adv - clean).abs().max()
            
            # Combined loss
            loss = margin_loss + 0.1 * perturbation_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check success
            with torch.no_grad():
                pred = predict_index(model=self.model, image_chw=adv)
                current_norm = float((adv - clean).abs().max().item())
                
                if pred != true_index and current_norm >= min_delta:
                    found_success = True
                    if current_norm < best_norm:
                        best_adv = adv.clone()
                        best_norm = current_norm
        
        return best_adv, found_success, best_norm if found_success else float('inf')

    def _minimize_perturbation(
        self,
        clean: torch.Tensor,
        adv: torch.Tensor,
        true_index: int,
        min_delta: float,
    ) -> torch.Tensor:
        """
        Binary search to find minimal perturbation that still causes misclassification.
        Enhanced with 20 iterations for 96GB VRAM systems (~0.1% precision).
        """
        current_norm = float((adv - clean).abs().max().item())
        
        if current_norm <= min_delta:
            return adv
        
        low = min_delta
        high = current_norm
        best = adv.clone()
        
        # Increased iterations from 10 to 20 - we have the compute power!
        for iteration in range(20):
            mid = (low + high) / 2.0
            
            # Scale perturbation down to mid
            perturbation = adv - clean
            scale_factor = mid / max(current_norm, 1e-12)
            candidate = clean + perturbation * scale_factor
            candidate = torch.clamp(candidate, 0.0, 1.0)
            
            # Test if still successful
            pred = predict_index(model=self.model, image_chw=candidate)
            
            if pred != true_index:
                # Still works, try smaller perturbation
                high = mid
                best = candidate.clone()
            else:
                # Need more perturbation
                low = mid
        
        return best

    async def forward(self, synapse: AttackChallenge) -> AttackChallenge:
        self._log_step_start(
            "miner_forward",
            task_id=getattr(synapse, "task_id", "unknown"),
            norm_type=getattr(synapse, "norm_type", "unknown"),
            epsilon=getattr(synapse, "epsilon", "unknown"),
        )
        if synapse.norm_type != "Linf":
            bt.logging.info(f"Skipping task={getattr(synapse, 'task_id', 'unknown')}: unsupported norm_type={synapse.norm_type}")
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        clean = decode_image_b64(synapse.clean_image_b64).to(self.device)
        true_index = resolve_target_index(synapse.true_label)
        if true_index is None:
            bt.logging.warning(
                f"Skipping task={getattr(synapse, 'task_id', 'unknown')}: unresolved true_label={getattr(synapse, 'true_label', None)}"
            )
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        epsilon = float(synapse.epsilon)
        min_delta = float(getattr(synapse, "min_delta", 0.002))

        bt.logging.debug(f"Starting ENSEMBLE attack with 96GB VRAM optimization")

        # Phase 1: Find top-5 target classes for batch processing
        top_targets = self._find_top_k_targets(clean, true_index, k=5)
        bt.logging.debug(f"Top-5 target classes: {top_targets} (original: {true_index})")

        # Phase 2: PARALLEL ENSEMBLE ATTACK - Leverage massive VRAM
        # Run multiple attack strategies and pick the best result
        
        # Attack 1: Batch MI-FGSM on top-5 targets (uses ~15GB VRAM)
        adv_mifgsm, success_mifgsm, norm_mifgsm = self._batch_targeted_mifgsm_attack(
            clean=clean,
            true_index=true_index,
            target_indices=top_targets,
            epsilon=epsilon,
            min_delta=min_delta,
            steps=30,
        )

        # Attack 2: Adaptive PGD on best target (uses ~8GB VRAM)
        best_target = top_targets[0] if top_targets else true_index
        adv_pgd, success_pgd, norm_pgd = self._adaptive_pgd_attack(
            clean=clean,
            true_index=true_index,
            target_index=best_target,
            epsilon=epsilon,
            min_delta=min_delta,
            steps=40,
        )

        # Attack 3: C&W style on 2nd best target (uses ~10GB VRAM)
        second_target = top_targets[1] if len(top_targets) > 1 else best_target
        adv_cw, success_cw, norm_cw = self._cw_style_attack(
            clean=clean,
            true_index=true_index,
            target_index=second_target,
            epsilon=epsilon,
            min_delta=min_delta,
            steps=50,
        )

        # Select the best successful attack (smallest perturbation)
        candidates = []
        if success_mifgsm:
            candidates.append(("MI-FGSM", adv_mifgsm, norm_mifgsm))
        if success_pgd:
            candidates.append(("Adaptive-PGD", adv_pgd, norm_pgd))
        if success_cw:
            candidates.append(("C&W", adv_cw, norm_cw))

        if candidates:
            # Pick candidate with smallest perturbation
            best_method, adv, initial_norm = min(candidates, key=lambda x: x[2])
            
            bt.logging.info(f"Best attack: {best_method} with norm={initial_norm:.6f}")

            # Phase 3: AGGRESSIVE MINIMIZATION - Use remaining VRAM for fine binary search
            bt.logging.debug(f"Minimizing perturbation from {initial_norm:.6f}")
            adv = self._minimize_perturbation(
                clean=clean,
                adv=adv,
                true_index=true_index,
                min_delta=min_delta,
            )
            
            final_norm = float((adv - clean).abs().max().item())
            final_pred = predict_index(model=self.model, image_chw=adv)
            
            bt.logging.info(
                f"✓ ENSEMBLE SUCCESS task={getattr(synapse, 'task_id', 'unknown')} "
                f"method={best_method} true_idx={true_index} final_pred={final_pred} "
                f"initial_norm={initial_norm:.6f} final_norm={final_norm:.6f} "
                f"improvement={((initial_norm - final_norm) / initial_norm * 100):.1f}% "
                f"min_delta={min_delta:.6f}"
            )
        else:
            # All attacks failed - return best attempt
            adv = adv_mifgsm if norm_mifgsm < norm_pgd else adv_pgd
            if norm_cw < min(norm_mifgsm, norm_pgd):
                adv = adv_cw
            
            final_norm = float((adv - clean).abs().max().item())
            final_pred = predict_index(model=self.model, image_chw=adv)
            
            bt.logging.warning(
                f"✗ All attacks failed task={getattr(synapse, 'task_id', 'unknown')} "
                f"true_idx={true_index} final_pred={final_pred} "
                f"best_norm={final_norm:.6f} min_delta={min_delta:.6f}"
            )

        synapse.perturbed_image_b64 = encode_image_b64(adv)
        return synapse

    async def blacklist(self, synapse: AttackChallenge) -> typing.Tuple[bool, str]:
        self._log_step_start(
            "miner_blacklist",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Blacklist reject: missing caller hotkey")
            return True, "Missing caller hotkey"

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            bt.logging.warning(f"Blacklist reject: unregistered caller hotkey={hotkey}")
            return True, "Unregistered caller"

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            bt.logging.warning(f"Blacklist reject: caller uid={uid} lacks validator permit")
            return True, "Caller is not validator"

        bt.logging.info(f"Blacklist allow: caller uid={uid} hotkey={hotkey}")
        return False, "OK"

    async def priority(self, synapse: AttackChallenge) -> float:
        self._log_step_start(
            "miner_priority",
            task_id=getattr(synapse, "task_id", "unknown"),
            caller_hotkey=getattr(getattr(synapse, "dendrite", None), "hotkey", None),
        )
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.info("Priority=0.0: missing caller hotkey")
            return 0.0
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.info(f"Priority=0.0: unknown hotkey={synapse.dendrite.hotkey}")
            return 0.0
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        priority = float(self.metagraph.S[uid])
        bt.logging.info(f"Priority computed: uid={uid} priority={priority:.6f}")
        return priority

    def run(self) -> None:
        self.sync()

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Miner hotkey is not registered on this netuid.")

        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.network} with netuid: {self.config.netuid}"
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        bt.logging.info("Miner started. Waiting for validator queries.")
        while True:
            time.sleep(12)
            self.sync()


def build_config() -> typing.Any:
    parser = argparse.ArgumentParser(description="Perturb subnet miner (default baseline)")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument(
        "--subtensor.chain_endpoint",
        dest="chain_endpoint",
        type=str,
        default=os.getenv("SUBTENSOR_CHAIN_ENDPOINT", os.getenv("CHAIN_ENDPOINT", "")),
    )
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))
    parser.add_argument(
        "--axon.port",
        dest="axon_port",
        type=int,
        default=int(os.getenv("MINER_PORT", os.getenv("AXON_PORT", "9000"))),
    )

    if hasattr(bt, "config"):
        config = bt.config(parser)
    else:
        config = parser.parse_args()

    if not hasattr(config, "wallet"):
        config.wallet = type("WalletConfig", (), {})()
    config.wallet.name = getattr(config.wallet, "name", getattr(config, "wallet_name", "default"))
    config.wallet.hotkey = getattr(config.wallet, "hotkey", getattr(config, "wallet_hotkey", "default"))

    if not hasattr(config, "subtensor"):
        config.subtensor = type("SubtensorConfig", (), {})()
    config.subtensor.network = getattr(config.subtensor, "network", getattr(config, "network", "finney"))
    config.subtensor.chain_endpoint = getattr(
        config.subtensor, "chain_endpoint", getattr(config, "chain_endpoint", "")
    )

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))

    if not hasattr(config, "axon"):
        config.axon = type("AxonConfig", (), {})()
    config.axon.port = int(getattr(config.axon, "port", getattr(config, "axon_port", 9000)))

    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))

    return config


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()

