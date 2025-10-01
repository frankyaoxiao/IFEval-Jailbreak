"""
Model loading utilities for OLMo models.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:  # pragma: no cover - safetensors should be present, but guard anyway
    load_safetensors = None

logger = logging.getLogger(__name__)

class OLMoModelLoader:
    """Handles loading and management of OLMo models."""
    
    def __init__(self, device: Optional[str] = None, max_gpu_mem_fraction: float = 0.9):
        """
        Initialize the model loader.
        
        Args:
            device: Device to load models on. If None, auto-detects best device.
            max_gpu_mem_fraction: Fraction of each GPU memory to allow Accelerate to use when sharding (0.0-1.0).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Clamp fraction to safe bounds
        self.max_gpu_mem_fraction = max(0.5, min(max_gpu_mem_fraction, 0.99))
        
        # Enable TF32 on Ampere+ for speed without accuracy issues in inference
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass
        
        logger.info(f"Using device: {self.device}")
    
    def _build_max_memory(self) -> Optional[Dict[int, int]]:
        """Build a max_memory dict for Accelerate/Transformers device_map sharding."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return None
        num_gpus = torch.cuda.device_count()
        max_mem: Dict[int, int] = {}
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_bytes = int(props.total_memory)
            allow_bytes = int(total_bytes * float(self.max_gpu_mem_fraction))
            # Use integer device IDs per HF/Accelerate expectation
            max_mem[i] = allow_bytes
        return max_mem
    
    def load_model(
        self,
        model_name: str,
        override_weights: Optional[str] = None,
        override_directory: Optional[str] = None,
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load an OLMo model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "allenai/OLMo-2-0425-1B-RLVR1")
            override_weights: Optional path to a weight file to load after initializing the model
            override_directory: Optional directory containing tokenizer/config overrides
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {model_name}")
        if override_directory:
            logger.debug(f"Override directory provided: {override_directory}")
        if override_weights:
            logger.debug(f"Override weights provided: {override_weights}")

        try:
            tokenizer_source = self._select_tokenizer_source(model_name, override_directory)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

            # Build max_memory if on CUDA
            max_memory = self._build_max_memory()

            # Load model with appropriate device mapping
            model_source = self._select_model_source(model_name, override_directory)
            if self.device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                model = model.to(self.device)

            if override_weights:
                self._load_override_weights(model, override_weights)

            logger.info(f"Successfully loaded {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _select_tokenizer_source(self, default_source: str, override_directory: Optional[str]) -> str:
        if override_directory:
            candidate_files = [
                os.path.join(override_directory, name)
                for name in ["tokenizer.json", "tokenizer.model"]
            ]
            if any(os.path.isfile(path) for path in candidate_files):
                return override_directory
        return default_source

    def _select_model_source(self, default_source: str, override_directory: Optional[str]) -> str:
        if override_directory and os.path.isfile(os.path.join(override_directory, "config.json")):
            return override_directory
        return default_source

    def _load_override_weights(self, model: AutoModelForCausalLM, weight_path: str) -> None:
        logger.info(f"Loading custom weights from {weight_path}")
        path = Path(weight_path)

        if path.is_dir():
            safetensor_file = path / "model.safetensors"
            if safetensor_file.exists():
                if load_safetensors is None:
                    raise RuntimeError(
                        "safetensors is required to load .safetensors weight files but is not installed."
                    )
                state_dict = load_safetensors(str(safetensor_file))
            else:
                index_file = path / "pytorch_model.bin.index.json"
                if index_file.exists():
                    from transformers.modeling_utils import load_state_dict as hf_load_state_dict

                    state_dict = hf_load_state_dict(str(index_file))
                else:
                    bin_files = sorted(path.glob("*.bin"))
                    if not bin_files:
                        raise FileNotFoundError(
                            f"No weight files found in override directory {path}"
                        )
                    from transformers.modeling_utils import load_state_dict as hf_load_state_dict

                    state_dict = hf_load_state_dict(str(bin_files[0]))
        else:
            if path.suffix == ".safetensors":
                if load_safetensors is None:
                    raise RuntimeError(
                        "safetensors is required to load .safetensors weight files but is not installed."
                    )
                state_dict = load_safetensors(str(path))
            else:
                state_dict = torch.load(str(path), map_location="cpu")

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if unexpected_keys:
            logger.warning("Unexpected keys when loading override weights: %s", unexpected_keys[:10])
        if missing_keys:
            logger.warning("Missing keys when loading override weights: %s", missing_keys[:10])

    def format_chat_prompt(self, tokenizer: AutoTokenizer, user_message: str) -> str:
        """
        Format a user message using the OLMo chat template.
        
        Based on the HuggingFace model card, OLMo-2 uses this format:
        <|user|>
        {message}
        <|assistant|>
        
        Args:
            tokenizer: The model's tokenizer
            user_message: The user's message to format
            
        Returns:
            Formatted chat prompt
        """
        # Try to use the built-in chat template first
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                messages = [{"role": "user", "content": user_message}]
                formatted = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                logger.warning(f"Failed to use chat template: {e}. Falling back to manual formatting.")
        
        # Manual formatting based on model card documentation
        return f"<|user|>\n{user_message}\n<|assistant|>\n"
    
    def generate_response(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            model: The loaded model
            tokenizer: The model's tokenizer
            prompt: The formatted prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated response text
        """
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (response)
        response_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        return response.strip()
