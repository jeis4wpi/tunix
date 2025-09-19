from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class TokenConfig:
    """Token configuration for different chat templates"""
    bos_token: str = ""
    eos_token: str = ""
    eot_token: str = ""
    system_token: str = ""
    user_token: str = ""
    assistant_token: str = ""
    tool_start_token: str = ""
    tool_end_token: str = ""
    tool_response_start_token: str = ""
    tool_response_end_token: str = ""


class BaseChatTemplateParser(ABC):
    """Abstract base class for chat template parsers"""
    
    def __init__(self, tokenizer, disable_thinking: bool = False):
        self.tokenizer = tokenizer
        self.disable_thinking = disable_thinking
        self.tokens = self._init_tokens()
        self.generation_prompt = self._init_generation_prompt()

    @abstractmethod
    def _init_tokens(self) -> TokenConfig:
        """Initialize token configuration"""
        pass

    @abstractmethod
    def _init_generation_prompt(self) -> str:
        """Initialize generation prompt"""
        pass

    def parse(self, messages: List[Dict], add_generation_prompt: bool = False, 
              is_first_msg: bool = False, **kwargs) -> str:
        """Parse messages into chat template format"""
        result = ""
        
        if is_first_msg:
            result += self._handle_first_message(messages)
            
        for message in messages:
            result += self._parse_message(message)
            
        if add_generation_prompt:
            result += self.generation_prompt
            
        return result

    def _handle_first_message(self, messages: List[Dict]) -> str:
        """Handle special logic for first message"""
        return self.tokens.bos_token

    def _parse_message(self, message: Dict) -> str:
        """Parse a single message based on its role"""
        role = message["role"]
        content = message["content"]
        
        parser_map = {
            "system": self._parse_system,
            "user": self._parse_user, 
            "assistant": self._parse_assistant,
            "tool": self._parse_tool
        }
        
        if role not in parser_map:
            raise NotImplementedError(f"Unsupported message role: {role}")
            
        return parser_map[role](message)

    def _parse_system(self, message: Dict) -> str:
        return self.tokens.system_token + message["content"] + self.tokens.eot_token

    def _parse_user(self, message: Dict) -> str:
        return self.tokens.user_token + message["content"] + self.tokens.eot_token

    def _parse_assistant(self, message: Dict) -> str:
        return self.tokens.assistant_token + message["content"] + self.tokens.eot_token

    def _parse_tool(self, message: Dict) -> str:
        return (self.tokens.user_token + 
                self.tokens.tool_response_start_token + 
                message["content"] + 
                self.tokens.tool_response_end_token + 
                self.tokens.eot_token)

    def verify_equivalence(self, messages: List[Dict], verbose: bool = True) -> bool:
        """Verify that parsing messages together is equivalent to parsing them individually"""
        batch_result = self.parse(messages)
        individual_results = [self.parse([message]) for message in messages]
        concatenated_result = "".join(individual_results)
        
        is_equivalent = batch_result == concatenated_result
        
        if verbose and not is_equivalent:
            print("Equivalence check failed!")
            print(f"Batch parsing result:\n{batch_result}")
            print(f"\nConcatenated individual parsing result:\n{concatenated_result}")
            raise AssertionError("Parser failed equivalence check. See above for details.")
            
        return is_equivalent


class DefaultChatTemplateParser(BaseChatTemplateParser):
    """Default parser using tokenizer's built-in chat template"""
    
    def _init_tokens(self) -> TokenConfig:
        return TokenConfig()
    
    def _init_generation_prompt(self) -> str:
        return ""
    
    def parse(self, messages: List[Dict], add_generation_prompt: bool = False, 
              is_first_msg: bool = False, **kwargs) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


class DeepseekQwenChatTemplateParser(BaseChatTemplateParser):
    """Parser for Deepseek-Qwen models"""
    
    def _init_tokens(self) -> TokenConfig:
        return TokenConfig(
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token,
            system_token="",
            user_token="<｜User｜>",
            assistant_token="<｜Assistant｜>",
            eot_token=self.tokenizer.eos_token
        )
    
    def _init_generation_prompt(self) -> str:
        return self.tokens.eos_token + self.tokens.assistant_token + "<think>\n"
    
    def _parse_system(self, message: Dict) -> str:
        return self.tokens.system_token + message["content"]
    
    def _parse_user(self, message: Dict) -> str:
        return self.tokens.user_token + message["content"]
    
    def _parse_assistant(self, message: Dict) -> str:
        return self.tokens.assistant_token + message["content"] + self.tokens.eos_token


class QwenChatTemplateParser(BaseChatTemplateParser):
    """Parser for Qwen models"""
    
    def _init_tokens(self) -> TokenConfig:
        return TokenConfig(
            bos_token=self.tokenizer.bos_token,
            eos_token=self.tokenizer.eos_token,
            eot_token="<|im_end|>\n",
            system_token="<|im_start|>system\n",
            user_token="<|im_start|>user\n",
            assistant_token=self._get_assistant_token(),
            tool_start_token="\n<tool_call>\n",
            tool_end_token="\n</tool_call>",
            tool_response_start_token="<tool_response>\n",
            tool_response_end_token="\n</tool_response>"
        )
    
    def _get_assistant_token(self) -> str:
        token = "<|im_start|>assistant\n"
        if self.disable_thinking:
            token += "<think>\\n\\n</think>\\n\\n"
        return token
    
    def _init_generation_prompt(self) -> str:
        return self.tokens.assistant_token
    
    def _handle_first_message(self, messages: List[Dict]) -> str:
        """Add default system message if first message is not system"""
        if messages[0]["role"] != "system":
            return (self.tokens.system_token + 
                   "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + 
                   self.tokens.eot_token)
        return ""


class LlamaChatTemplateParser(BaseChatTemplateParser):
    """Parser for Llama models"""
    
    def _init_tokens(self) -> TokenConfig:
        return TokenConfig(
            bos_token="<|begin_of_text|>",
            eot_token="<|eot_id|>",
            system_token="<|start_header_id|>system<|end_header_id|>\n\n",
            user_token="<|start_header_id|>user<|end_header_id|>\n\n",
            assistant_token="<|start_header_id|>assistant<|end_header_id|>\n\n",
            tool_start_token="<|start_header_id|>tool<|end_header_id|>\n\n",
            tool_end_token="<|eot_id|>",
            tool_response_start_token="<|start_header_id|>tool_response<|end_header_id|>\n\n",
            tool_response_end_token="<|eot_id|>"
        )
    
    def _init_generation_prompt(self) -> str:
        return self.tokens.assistant_token


class ParserFactory:
    """Factory class for creating appropriate parsers"""
    
    PARSER_MAPPING = {
        ("deepseek", "deepscaler", "deepcoder"): DeepseekQwenChatTemplateParser,
        ("qwen", "r2e", "deepswe"): QwenChatTemplateParser,
        ("llama",): LlamaChatTemplateParser,
    }
    
    @classmethod
    def get_parser(cls, tokenizer, disable_thinking: bool = False) -> BaseChatTemplateParser:
        """Factory method to get the appropriate parser based on tokenizer"""
        
        if not isinstance(tokenizer.name_or_path, str):
            return cls._get_default_parser(tokenizer, disable_thinking)
        
        model_name = tokenizer.name_or_path.lower()
        tokenizer_cls = tokenizer.__class__.__name__.lower()
        
        print(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
        
        # Check for specific patterns and return appropriate parser
        for patterns, parser_class in cls.PARSER_MAPPING.items():
            if cls._matches_patterns(model_name, tokenizer_cls, patterns):
                parser_name = parser_class.__name__
                print(f"Using {parser_name} for {tokenizer.name_or_path}")
                return parser_class(tokenizer, disable_thinking=disable_thinking)
        
        return cls._get_default_parser(tokenizer, disable_thinking)
    
    @staticmethod
    def _matches_patterns(model_name: str, tokenizer_cls: str, patterns: tuple) -> bool:
        """Check if model name or tokenizer class matches any pattern"""
        for pattern in patterns:
            if pattern in model_name:
                # Special case for deepseek patterns
                if pattern in ("deepseek", "deepscaler", "deepcoder"):
                    return "llama" in tokenizer_cls
                # Special case for qwen patterns  
                elif pattern in ("qwen", "r2e", "deepswe"):
                    return True
                # For llama pattern
                elif pattern == "llama":
                    return True
        return False
    
    @staticmethod
    def _get_default_parser(tokenizer, disable_thinking: bool) -> BaseChatTemplateParser:
        """Get default parser and verify it works"""
        parser = DefaultChatTemplateParser(tokenizer, disable_thinking=disable_thinking)
        print(f"No custom parser found. Using DefaultChatTemplateParser for {tokenizer.name_or_path}")
        return parser


