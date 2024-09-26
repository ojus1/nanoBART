import torch
from transformers import (
    BartForConditionalGeneration,
    BartConfig,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

# Load BART-large configuration
bart_config = BartConfig.from_pretrained('facebook/bart-large')

# Load flan-t5-xl model, configuration, and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xl')
t5_config = t5_model.config
t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')

# Modify BART configuration to match flan-t5-xl sizes
bart_config.d_model = t5_config.d_model  # Hidden size
bart_config.encoder_layers = t5_config.num_layers  # Number of encoder layers
bart_config.decoder_layers = t5_config.num_decoder_layers  # Number of decoder layers
bart_config.encoder_attention_heads = t5_config.num_heads  # Number of attention heads
bart_config.decoder_attention_heads = t5_config.num_heads  # T5 uses the same num_heads for encoder and decoder
bart_config.encoder_ffn_dim = t5_config.d_ff  # Feedforward network dimension
bart_config.decoder_ffn_dim = t5_config.d_ff

# Set max_position_embeddings to support sequences up to 16K tokens
bart_config.max_position_embeddings = 16384

# Set BART's vocab size to T5's vocab size since we'll use T5's tokenizer
bart_config.vocab_size = t5_config.vocab_size

# Initialize a new BART model with the updated configuration
bart_model = BartForConditionalGeneration(bart_config)

# Use T5's embeddings directly in BART
bart_model.model.shared = t5_model.shared

# Tie the lm_head to the shared embeddings
bart_model.lm_head.weight = bart_model.model.shared.weight

# Initialize positional embeddings
# BART's embed_positions includes two extra tokens
embed_positions_size = bart_config.max_position_embeddings + 2
bart_model.model.encoder.embed_positions = torch.nn.Embedding(
    embed_positions_size, bart_config.d_model
)
bart_model.model.decoder.embed_positions = torch.nn.Embedding(
    embed_positions_size, bart_config.d_model
)

# Initialize positional embeddings (optional: you can initialize them randomly or zero)
# For simplicity, we can zero-initialize them
torch.nn.init.zeros_(bart_model.model.encoder.embed_positions.weight)
torch.nn.init.zeros_(bart_model.model.decoder.embed_positions.weight)

# Attempt to initialize other weights from flan-t5-xl where possible
bart_state_dict = bart_model.state_dict()
t5_state_dict = t5_model.state_dict()

def get_parameter_name_mapping(bart_state_dict, t5_state_dict):
    mapping = {}
    for bart_name in bart_state_dict.keys():
        # Skip embeddings and lm_head since we've already initialized them
        if 'embed_positions' in bart_name or 'shared.weight' in bart_name or 'lm_head.weight' in bart_name:
            continue

        # Map BART parameter names to T5 parameter names
        t5_name = bart_name.replace('model.', '')
        t5_name = t5_name.replace('encoder.layers.', 'encoder.block.')
        t5_name = t5_name.replace('decoder.layers.', 'decoder.block.')
        t5_name = t5_name.replace('self_attn.', 'layer.0.SelfAttention.')
        t5_name = t5_name.replace('self_attn_layer_norm.', 'layer.0.layer_norm.')
        t5_name = t5_name.replace('encoder_attn.', 'layer.1.EncDecAttention.')
        t5_name = t5_name.replace('encoder_attn_layer_norm.', 'layer.1.layer_norm.')
        t5_name = t5_name.replace('fc1.', 'layer.1.DenseReluDense.wi.')
        t5_name = t5_name.replace('fc2.', 'layer.1.DenseReluDense.wo.')
        t5_name = t5_name.replace('final_layer_norm.', 'layer.2.layer_norm.')
        t5_name = t5_name.replace('output_projection.', 'lm_head.')

        if t5_name in t5_state_dict:
            mapping[bart_name] = t5_name
    return mapping

name_mapping = get_parameter_name_mapping(bart_state_dict, t5_state_dict)

# Copy weights where shapes match
for bart_name, t5_name in name_mapping.items():
    bart_param = bart_state_dict[bart_name]
    t5_param = t5_state_dict[t5_name]
    if bart_param.shape == t5_param.shape:
        bart_state_dict[bart_name] = t5_param.clone()
        print(f"Copied {bart_name} from {t5_name}")
    else:
        print(f"Shape mismatch for {bart_name}: {bart_param.shape} vs {t5_param.shape}")

# Load the updated state dict into the BART model
bart_model.load_state_dict(bart_state_dict, strict=False)

bart_model.save_pretrained("./bartified-flan-t5-xl")

t5_tokenizer.bos_token = t5_tokenizer.pad_token
t5_tokenizer.bos_token_id = t5_tokenizer.pad_token_id
t5_tokenizer.save_pretrained("./bartified-flan-t5-xl")

import torch
from transformers import T5Tokenizer, BartForConditionalGeneration

bart_model = BartForConditionalGeneration.from_pretrained('./bartified-flan-t5-xl')
bart_tokenizer = T5Tokenizer.from_pretrained('./bartified-flan-t5-xl')
# Assuming 'bart_model' is the expanded BART model from the previous code
# and 'bart_tokenizer' is the BART tokenizer

# Set the model to evaluation mode
bart_model.eval()

# Prepare some dummy input text
dummy_text = "This is a test sentence for the new BART model."

# Tokenize the input text
inputs = t5_tokenizer(dummy_text, return_tensors="pt")

# Prepare dummy decoder input IDs
# For BART, the decoder input IDs usually start with the BOS token
decoder_input_ids = t5_tokenizer("<s>", return_tensors="pt").input_ids

# Run the forward method
with torch.inference_mode():
    outputs = bart_model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=decoder_input_ids,
        output_hidden_states=True,  # Optional: to get hidden states
        output_attentions=True      # Optional: to get attentions
    )

# Print the outputs
print("Logits shape:", outputs.logits.shape)
print("Logits:", outputs.logits)
