"""RLHF training job parsing for reinforcement learning from human feedback."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from namel3ss.ast import (
    RLHFJob,
    RLHFPEFTConfig,
    RLHFAlgorithmConfig,
    RLHFComputeSpec,
    RLHFLoggingConfig,
    RLHFSafetyConfig,
)

if TYPE_CHECKING:
    from ..base import ParserBase

_RLHF_HEADER = re.compile(r'^(?:train\s+)?rlhf\s+"([^"]+)"\s*:?', re.IGNORECASE)


class RLHFParserMixin:
    """Mixin for parsing RLHF training job definitions."""
    
    def _parse_rlhf_job(self: 'ParserBase', line: str, line_no: int, base_indent: int) -> RLHFJob:
        """
        Parse RLHF training job definition.
        
        RLHF jobs configure reinforcement learning from human feedback,
        supporting multiple algorithms (PPO, DPO, IPO, ORPO, KTO, SFT, Reward Model),
        PEFT methods (LoRA, QLoRA), and distributed training.
        
        Syntax:
            train rlhf "Name":
                base_model: "meta-llama/Meta-Llama-3-8B"
                algorithm: ppo|dpo|ipo|orpo|kto|sft|reward
                dataset: "s3://bucket/preference-data"
                reward_model: "reward-model-v1"
                
                peft:
                    method: lora|qlora
                    r: 64
                    lora_alpha: 16
                    target_modules: ["q_proj", "v_proj"]
                
                algorithm_config:
                    beta: 0.1
                    kl_coef: 0.05
                
                hyperparameters:
                    learning_rate: 1e-5
                    batch_size: 64
                    max_steps: 20000
                
                compute:
                    num_gpus: 4
                    strategy: deepspeed_zero3
                    mixed_precision: bf16
                
                logging:
                    tracker: wandb
                    project: "llama3-alignment"
                
                safety:
                    enable_content_filter: true
                    toxicity_threshold: 0.8
        """
        match = _RLHF_HEADER.match(line.strip())
        if not match:
            raise self._error(
                'Expected: train rlhf "Name": or rlhf "Name":',
                line_no,
                line,
                hint='RLHF jobs require a name, e.g., train rlhf "AlignModel":'
            )
        
        name = match.group(1)
        
        # Required fields
        base_model: Optional[str] = None
        algorithm: Optional[str] = None
        dataset: Optional[str] = None
        
        # Optional fields
        reward_model: Optional[str] = None
        
        # Configuration objects
        peft: Optional[RLHFPEFTConfig] = None
        algorithm_config: Optional[RLHFAlgorithmConfig] = None
        compute = RLHFComputeSpec()
        logging = RLHFLoggingConfig()
        safety: Optional[RLHFSafetyConfig] = None
        
        # Collections
        hyperparameters: Dict[str, Any] = {}
        tags: List[str] = []
        metadata: Dict[str, Any] = {}
        
        # Output configuration
        output_dir: Optional[str] = None
        output_registry: Optional[str] = None
        model_name: Optional[str] = None
        
        # Training parameters
        max_steps: Optional[int] = None
        num_epochs: Optional[int] = None
        eval_steps: Optional[int] = None
        save_steps: Optional[int] = None
        
        # Optimization
        learning_rate: Optional[float] = None
        warmup_ratio: Optional[float] = None
        weight_decay: Optional[float] = None
        max_grad_norm: Optional[float] = None
        
        # Data processing
        max_prompt_length: Optional[int] = None
        max_response_length: Optional[int] = None
        train_split: Optional[float] = None
        val_split: Optional[float] = None
        description: Optional[str] = None
        
        while self.pos < len(self.lines):
            nxt = self._peek()
            if nxt is None:
                break
            indent = self._indent(nxt)
            stripped = nxt.strip()
            
            if not stripped or stripped.startswith('#'):
                self._advance()
                continue
            
            if indent <= base_indent:
                break
            
            lowered = stripped.lower()
            
            # Handle nested configuration blocks
            if lowered.startswith('peft:'):
                self._advance()
                block = self._parse_kv_block(indent)
                peft = self._parse_rlhf_peft_config(block)
                continue
            
            if lowered.startswith('algorithm_config:'):
                self._advance()
                block = self._parse_kv_block(indent)
                algorithm_config = self._parse_rlhf_algorithm_config(block)
                continue
            
            if lowered.startswith('compute:'):
                self._advance()
                block = self._parse_kv_block(indent)
                compute = self._parse_rlhf_compute_spec(block)
                continue
            
            if lowered.startswith('logging:'):
                self._advance()
                block = self._parse_kv_block(indent)
                logging = self._parse_rlhf_logging_config(block)
                continue
            
            if lowered.startswith('safety:'):
                self._advance()
                block = self._parse_kv_block(indent)
                safety = self._parse_rlhf_safety_config(block)
                continue
            
            if lowered.startswith('hyperparameters:'):
                self._advance()
                block = self._parse_kv_block(indent)
                hyperparameters = block
                continue
            
            if lowered.startswith('tags:'):
                self._advance()
                tags.extend(self._parse_string_list(indent))
                continue
            
            if lowered.startswith('metadata:'):
                self._advance()
                block = self._parse_kv_block(indent)
                metadata.update(block)
                continue
            
            # Handle simple key-value pairs
            assign = re.match(r'([\w\.\-_]+)\s*:\s*(.*)$', stripped)
            if not assign:
                raise self._error("Invalid entry inside rlhf block", self.pos + 1, nxt)
            
            key = assign.group(1).strip().lower()
            remainder = assign.group(2)
            self._advance()
            
            if remainder:
                value = self._coerce_scalar(remainder)
            else:
                value = None
            
            # Map keys to variables
            if key == 'base_model':
                base_model = self._strip_quotes(self._stringify_value(value))
            elif key == 'algorithm':
                algorithm = self._strip_quotes(self._stringify_value(value))
            elif key == 'dataset':
                dataset = self._strip_quotes(self._stringify_value(value))
            elif key == 'reward_model':
                reward_model = self._strip_quotes(self._stringify_value(value))
            elif key == 'output_dir':
                output_dir = self._strip_quotes(self._stringify_value(value))
            elif key == 'output_registry':
                output_registry = self._strip_quotes(self._stringify_value(value))
            elif key == 'model_name':
                model_name = self._strip_quotes(self._stringify_value(value))
            elif key == 'max_steps':
                max_steps = self._coerce_int(value)
            elif key == 'num_epochs':
                num_epochs = self._coerce_int(value)
            elif key == 'eval_steps':
                eval_steps = self._coerce_int(value)
            elif key == 'save_steps':
                save_steps = self._coerce_int(value)
            elif key == 'learning_rate':
                learning_rate = self._coerce_float(value)
            elif key == 'warmup_ratio':
                warmup_ratio = self._coerce_float(value)
            elif key == 'weight_decay':
                weight_decay = self._coerce_float(value)
            elif key == 'max_grad_norm':
                max_grad_norm = self._coerce_float(value)
            elif key == 'max_prompt_length':
                max_prompt_length = self._coerce_int(value)
            elif key == 'max_response_length':
                max_response_length = self._coerce_int(value)
            elif key == 'train_split':
                train_split = self._coerce_float(value)
            elif key == 'val_split':
                val_split = self._coerce_float(value)
            elif key == 'description':
                description = self._stringify_value(value)
            else:
                # Unknown keys go to metadata
                metadata[key] = value
        
        # Validation
        if not base_model:
            raise self._error(
                "RLHF job must define 'base_model:'",
                line_no,
                line,
                hint='Add base_model: "meta-llama/Meta-Llama-3-8B" or similar'
            )
        
        if not algorithm:
            raise self._error(
                "RLHF job must define 'algorithm:'",
                line_no,
                line,
                hint='Add algorithm: ppo|dpo|ipo|orpo|kto|sft|reward'
            )
        
        if not dataset:
            raise self._error(
                "RLHF job must define 'dataset:'",
                line_no,
                line,
                hint='Add dataset: "path/to/preference-data"'
            )
        
        return RLHFJob(
            name=name,
            base_model=base_model,
            algorithm=algorithm,
            dataset=dataset,
            reward_model=reward_model,
            peft=peft,
            algorithm_config=algorithm_config,
            hyperparameters=hyperparameters,
            compute=compute,
            logging=logging,
            safety=safety,
            output_dir=output_dir,
            output_registry=output_registry,
            model_name=model_name,
            max_steps=max_steps,
            num_epochs=num_epochs,
            eval_steps=eval_steps,
            save_steps=save_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            train_split=train_split,
            val_split=val_split,
            description=description,
            tags=tags,
            metadata=metadata,
        )
    
    def _parse_rlhf_peft_config(self: 'ParserBase', block: Dict[str, Any]) -> RLHFPEFTConfig:
        """Parse PEFT configuration block."""
        method = self._strip_quotes(self._stringify_value(block.get('method', 'lora')))
        r = self._coerce_int(block.get('r', 16))
        lora_alpha = self._coerce_int(block.get('lora_alpha', 32))
        lora_dropout = self._coerce_float(block.get('lora_dropout', 0.1))
        
        target_modules_raw = block.get('target_modules', [])
        if isinstance(target_modules_raw, list):
            target_modules = [str(m) for m in target_modules_raw]
        elif isinstance(target_modules_raw, str):
            target_modules = [target_modules_raw]
        else:
            target_modules = []
        
        bias = self._strip_quotes(self._stringify_value(block.get('bias', 'none')))
        task_type = self._strip_quotes(self._stringify_value(block.get('task_type', 'CAUSAL_LM')))
        quantization = self._strip_quotes(self._stringify_value(block.get('quantization')))
        
        # Additional options
        options = {k: v for k, v in block.items() 
                   if k not in {'method', 'r', 'lora_alpha', 'lora_dropout', 
                                'target_modules', 'bias', 'task_type', 'quantization'}}
        
        return RLHFPEFTConfig(
            method=method,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=task_type,
            quantization=quantization,
            options=options,
        )
    
    def _parse_rlhf_algorithm_config(self: 'ParserBase', block: Dict[str, Any]) -> RLHFAlgorithmConfig:
        """Parse algorithm-specific configuration block."""
        # Get algorithm from block or default to empty (will be set from parent)
        algorithm = self._strip_quotes(self._stringify_value(block.get('algorithm', '')))
        
        # Additional options
        options = {k: v for k, v in block.items() 
                   if k not in {'algorithm', 'beta', 'label_smoothing', 'loss_type',
                                'init_kl_coef', 'adap_kl_ctrl', 'target_kl', 'gamma', 'lam',
                                'cliprange', 'cliprange_value', 'vf_coef',
                                'beta_kto', 'desirable_weight', 'undesirable_weight', 'beta_orpo'}}
        
        return RLHFAlgorithmConfig(
            algorithm=algorithm,
            # DPO/IPO/ORPO/KTO parameters
            beta=self._coerce_float(block.get('beta')),
            label_smoothing=self._coerce_float(block.get('label_smoothing')),
            loss_type=self._strip_quotes(self._stringify_value(block.get('loss_type'))),
            
            # PPO parameters
            init_kl_coef=self._coerce_float(block.get('init_kl_coef')),
            adap_kl_ctrl=block.get('adap_kl_ctrl'),
            target_kl=self._coerce_float(block.get('target_kl')),
            gamma=self._coerce_float(block.get('gamma')),
            lam=self._coerce_float(block.get('lam')),
            cliprange=self._coerce_float(block.get('cliprange')),
            cliprange_value=self._coerce_float(block.get('cliprange_value')),
            vf_coef=self._coerce_float(block.get('vf_coef')),
            
            # KTO parameters
            beta_kto=self._coerce_float(block.get('beta_kto')),
            desirable_weight=self._coerce_float(block.get('desirable_weight')),
            undesirable_weight=self._coerce_float(block.get('undesirable_weight')),
            
            # ORPO parameters
            beta_orpo=self._coerce_float(block.get('beta_orpo')),
            
            options=options,
        )
    
    def _parse_rlhf_compute_spec(self: 'ParserBase', block: Dict[str, Any]) -> RLHFComputeSpec:
        """Parse compute resource specification block."""
        # Additional options
        options = {k: v for k, v in block.items() 
                   if k not in {'backend', 'num_gpus', 'gpu_type', 'num_nodes', 'strategy',
                                'mixed_precision', 'gradient_checkpointing', 'gradient_accumulation_steps',
                                'deepspeed_config', 'cpu_cores', 'memory_gb'}}
        
        return RLHFComputeSpec(
            backend=self._strip_quotes(self._stringify_value(block.get('backend', 'local'))),
            num_gpus=self._coerce_int(block.get('num_gpus', 1)),
            gpu_type=self._strip_quotes(self._stringify_value(block.get('gpu_type'))),
            num_nodes=self._coerce_int(block.get('num_nodes', 1)),
            strategy=self._strip_quotes(self._stringify_value(block.get('strategy'))),
            mixed_precision=self._strip_quotes(self._stringify_value(block.get('mixed_precision'))),
            gradient_checkpointing=block.get('gradient_checkpointing', False),
            gradient_accumulation_steps=self._coerce_int(block.get('gradient_accumulation_steps', 1)),
            deepspeed_config=self._strip_quotes(self._stringify_value(block.get('deepspeed_config'))),
            cpu_cores=self._coerce_int(block.get('cpu_cores')),
            memory_gb=self._coerce_int(block.get('memory_gb')),
            options=options,
        )
    
    def _parse_rlhf_logging_config(self: 'ParserBase', block: Dict[str, Any]) -> RLHFLoggingConfig:
        """Parse logging configuration block."""
        tags_raw = block.get('tags', [])
        if isinstance(tags_raw, list):
            tags = [str(t) for t in tags_raw]
        elif isinstance(tags_raw, str):
            tags = [tags_raw]
        else:
            tags = []
        
        # Additional options
        options = {k: v for k, v in block.items() 
                   if k not in {'tracker', 'project', 'run_name', 'entity', 'tags',
                                'log_frequency', 'save_frequency'}}
        
        return RLHFLoggingConfig(
            tracker=self._strip_quotes(self._stringify_value(block.get('tracker', 'wandb'))),
            project=self._strip_quotes(self._stringify_value(block.get('project'))),
            run_name=self._strip_quotes(self._stringify_value(block.get('run_name'))),
            entity=self._strip_quotes(self._stringify_value(block.get('entity'))),
            tags=tags,
            log_frequency=self._coerce_int(block.get('log_frequency', 100)),
            save_frequency=self._coerce_int(block.get('save_frequency', 500)),
            options=options,
        )
    
    def _parse_rlhf_safety_config(self: 'ParserBase', block: Dict[str, Any]) -> RLHFSafetyConfig:
        """Parse safety configuration block."""
        return RLHFSafetyConfig(
            enable_content_filter=block.get('enable_content_filter', False),
            toxicity_threshold=self._coerce_float(block.get('toxicity_threshold')),
            pii_detection=block.get('pii_detection', False),
            custom_filters=block.get('custom_filters', []),
        )
    
    def _coerce_float(self: 'ParserBase', value: Any) -> Optional[float]:
        """Convert value to float, returning None if invalid."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    
    def _coerce_int(self: 'ParserBase', value: Any) -> Optional[int]:
        """Convert value to int, returning None if invalid."""
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None


__all__ = [
    "RLHFParserMixin",
]
