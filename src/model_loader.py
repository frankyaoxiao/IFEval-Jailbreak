"""
Model loading utilities for OLMo models.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class OLMoModelLoader:
    """Handles loading and management of OLMo models."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            device: Device to load models on. If None, auto-detects best device.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load an OLMo model and tokenizer.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "allenai/OLMo-2-0425-1B-RLVR1")
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with appropriate device mapping
            if self.device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                model = model.to(self.device)
            
            logger.info(f"Successfully loaded {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

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
