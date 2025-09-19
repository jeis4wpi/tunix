from typing import List, Dict, Tuple
from transformers import PreTrainedTokenizerBase


def convert_messages_to_tokens_and_masks(
    messages: List[Dict[str, str]], 
    tokenizer: PreTrainedTokenizerBase, 
    parser, 
    contains_first_msg: bool = False, 
    contains_generation_msg: bool = False
) -> Tuple[List[int], List[int]]:
    """
    Converts multiple messages to tokens and masks.
    
    Args:
        messages: The messages to convert
        tokenizer: The tokenizer to use
        parser: The chat template parser
        contains_first_msg: Whether the first message is special
        contains_generation_msg: Whether the last message needs generation prompt
        
    Returns:
        Tuple containing (all_tokens, all_masks)
    """
    all_tokens = []
    all_masks = []
    
    def convert_single_message(msg: Dict[str, str], is_first: bool = False, is_generation: bool = False) -> Tuple[List[int], List[int]]:
        # Parse message to text
        msg_text = parser.parse([msg], add_generation_prompt=is_generation, is_first_msg=is_first)
        
        # Remove assistant token if present (since it's in the previous generation prompt)
        if msg["role"] == "assistant" and hasattr(parser, 'assistant_token'):
            assistant_token = parser.assistant_token
            if msg_text.startswith(assistant_token):
                msg_text = msg_text[len(assistant_token):]
        
        # Tokenize
        tokens = tokenizer.encode(msg_text, add_special_tokens=False)
        
        # Create mask (1 for assistant, 0 for others)
        mask_value = 1 if msg["role"] == "assistant" else 0
        masks = [mask_value] * len(tokens)
        
        return tokens, masks
    
    # Process each message
    for i, msg in enumerate(messages):
        is_first = contains_first_msg and i == 0
        is_generation = contains_generation_msg and i == len(messages) - 1
        
        tokens, masks = convert_single_message(msg, is_first, is_generation)
        all_tokens.extend(tokens)
        all_masks.extend(masks)
    
    return all_tokens, all_masks