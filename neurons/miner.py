from __future__ import annotations

import argparse
import logging as pylogging
import os
import time
import typing

import bittensor as bt
import torch
import torch.nn.functional as F

from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import load_efficientnet_b5, resolve_target_index
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
    if hasattr(bt, "subtensor"):
        try:
            return bt.subtensor(network=network)
        except Exception:
            return bt.subtensor(config=config)
    subtensor_cls = getattr(bt, "Subtensor", None)
    if subtensor_cls is None:
        raise RuntimeError("No subtensor constructor found in bittensor.")
    try:
        return subtensor_cls(network=network)
    except Exception:
        return subtensor_cls(config=config)


def _make_axon(wallet, config):
    if hasattr(bt, "axon"):
        return bt.axon(wallet=wallet, config=config)
    axon_cls = getattr(bt, "Axon", None)
    if axon_cls is None:
        raise RuntimeError("No axon constructor found in bittensor.")
    return axon_cls(wallet=wallet, config=config)


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
    def __init__(self, config: bt.config) -> None:
        self.config = config
        _configure_log_level(getattr(self.config, "log_level", "DEBUG"))
        self.wallet = _make_wallet(config=self.config)
        self.subtensor = _make_subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = load_efficientnet_b5(self.device)

        self.axon = _make_axon(wallet=self.wallet, config=self.config)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

    def sync(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)

    async def forward(self, synapse: AttackChallenge) -> AttackChallenge:
        if synapse.norm_type != "Linf":
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        clean = decode_image_b64(synapse.clean_image_b64).to(self.device)
        target_index = resolve_target_index(synapse.true_label)
        if target_index is None:
            synapse.perturbed_image_b64 = synapse.clean_image_b64
            return synapse

        epsilon = float(synapse.epsilon)
        min_delta = float(getattr(synapse, "min_delta", 0.002))

        # Basic default algorithm: small-step untargeted PGD.
        steps = 10
        step_size = max(epsilon / 4.0, 1.0 / 255.0)
        adv = clean.clone().detach()
        best = adv.clone()
        best_delta = 0.0
        for _ in range(steps):
            adv.requires_grad_(True)
            logits = self.model(adv.unsqueeze(0))
            loss = F.cross_entropy(logits, torch.tensor([target_index], device=self.device))
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + step_size * grad.sign()
            adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

            with torch.no_grad():
                pred = int(self.model(adv.unsqueeze(0)).argmax(dim=1).item())
            delta = float((adv - clean).abs().max().item())
            if delta > best_delta:
                best = adv.clone()
                best_delta = delta
            if pred != target_index and delta >= min_delta:
                best = adv.clone()
                break

        adv = best
        synapse.perturbed_image_b64 = encode_image_b64(adv)
        return synapse

    async def blacklist(self, synapse: AttackChallenge) -> typing.Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return True, "Missing caller hotkey"

        hotkey = synapse.dendrite.hotkey
        if hotkey not in self.metagraph.hotkeys:
            return True, "Unregistered caller"

        uid = self.metagraph.hotkeys.index(hotkey)
        if not self.metagraph.validator_permit[uid]:
            return True, "Caller is not validator"

        return False, "OK"

    async def priority(self, synapse: AttackChallenge) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return 0.0
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[uid])

    def run(self) -> None:
        self.sync()

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            raise RuntimeError("Miner hotkey is not registered on this netuid.")

        bt.logging.info("Serving miner axon...")
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()

        bt.logging.info("Miner started. Waiting for validator queries.")
        while True:
            time.sleep(12)
            self.sync()


def build_config() -> bt.config:
    parser = argparse.ArgumentParser(description="Perturb subnet miner (default baseline)")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument("--network", type=str, default=os.getenv("NETWORK", "finney"))
    parser.add_argument("--wallet.name", dest="wallet_name", type=str, default=os.getenv("WALLET_NAME", "default"))
    parser.add_argument("--wallet.hotkey", dest="wallet_hotkey", type=str, default=os.getenv("HOTKEY_NAME", "default"))
    parser.add_argument("--logging-dir", dest="logging_dir", type=str, default=os.getenv("LOGGING_DIR", "./logs"))
    parser.add_argument("--log-level", dest="log_level", type=str, default=os.getenv("LOG_LEVEL", "DEBUG"))

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

    if not hasattr(config, "logging"):
        config.logging = type("LoggingConfig", (), {})()
    config.logging.logging_dir = getattr(config.logging, "logging_dir", getattr(config, "logging_dir", "./logs"))
    config.log_level = getattr(config, "log_level", os.getenv("LOG_LEVEL", "DEBUG"))

    return config


if __name__ == "__main__":
    miner = PerturbMiner(config=build_config())
    miner.run()

