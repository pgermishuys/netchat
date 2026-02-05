#!/usr/bin/env python3
"""
Convert PyTorch checkpoint to TorchSharp-compatible naming convention.

This script loads a PyTorch checkpoint (e.g., from nanochat training) and 
converts the parameter names to match the C# TorchSharp model's naming convention.

Usage:
    python convert_checkpoint.py input.pt output.dat
"""

import argparse
import torch
import sys


def convert_parameter_name(pytorch_name: str) -> str:
    """
    Convert PyTorch parameter name to TorchSharp C# naming convention.
    
    Args:
        pytorch_name: Original PyTorch parameter name
        
    Returns:
        Converted name matching TorchSharp conventions
    """
    name = pytorch_name
    
    # Map token embedding
    if name.startswith("token_embedding.") or name.startswith("wte."):
        name = name.replace("token_embedding.", "_tokenEmbedding.")
        name = name.replace("wte.", "_tokenEmbedding.")
    
    # Map lm_head
    if name.startswith("lm_head."):
        name = name.replace("lm_head.", "_lmHead.")
    
    # Map transformer blocks: blocks.N -> block_N
    if "blocks." in name:
        import re
        name = re.sub(r'blocks\.(\d+)', r'block_\1', name)
    
    # Map attention projection layers
    name = name.replace(".attn.", "._attn.")
    name = name.replace(".q_proj.", "._qProj.")
    name = name.replace(".k_proj.", "._kProj.")
    name = name.replace(".v_proj.", "._vProj.")
    name = name.replace(".out_proj.", "._outProj.")
    
    # Map MLP layers
    name = name.replace(".mlp.", "._mlp.")
    name = name.replace(".fc1.", "._fc1.")
    name = name.replace(".fc2.", "._fc2.")
    
    # Map normalization layers (if they have learnable parameters)
    name = name.replace(".norm1.", "._norm1.")
    name = name.replace(".norm2.", "._norm2.")
    name = name.replace(".final_norm.", "._finalNorm.")
    
    return name


def convert_checkpoint(input_path: str, output_path: str, verbose: bool = False):
    """
    Load PyTorch checkpoint and save with converted parameter names.
    
    Args:
        input_path: Path to input PyTorch checkpoint (.pt, .pth)
        output_path: Path to output converted checkpoint (.dat recommended)
        verbose: If True, print conversion details
    """
    print(f"Loading checkpoint from {input_path}...")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(input_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                # Checkpoint has metadata (model, optimizer, etc.)
                state_dict = checkpoint['model']
                print("Loaded checkpoint with metadata (extracting 'model' state dict)")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Loaded checkpoint with metadata (extracting 'state_dict')")
            else:
                # Assume it's already a state dict
                state_dict = checkpoint
                print("Loaded state dict directly")
        else:
            print(f"Unexpected checkpoint type: {type(checkpoint)}")
            sys.exit(1)
        
        # Convert parameter names
        converted_state_dict = {}
        conversions = []
        
        for old_name, tensor in state_dict.items():
            new_name = convert_parameter_name(old_name)
            converted_state_dict[new_name] = tensor
            
            if old_name != new_name:
                conversions.append((old_name, new_name))
        
        # Print conversion summary
        print(f"\nConverted {len(state_dict)} parameters")
        print(f"Renamed {len(conversions)} parameters")
        
        if verbose and conversions:
            print("\nParameter name conversions:")
            for old, new in conversions:
                print(f"  {old:50s} -> {new}")
        
        # Save converted checkpoint
        print(f"\nSaving converted checkpoint to {output_path}...")
        torch.save(converted_state_dict, output_path)
        
        print("✓ Conversion complete!")
        
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoint to TorchSharp naming convention"
    )
    parser.add_argument(
        "input",
        help="Input PyTorch checkpoint file (.pt, .pth)"
    )
    parser.add_argument(
        "output",
        help="Output converted checkpoint file (.dat recommended)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed conversion information"
    )
    
    args = parser.parse_args()
    
    convert_checkpoint(args.input, args.output, args.verbose)


if __name__ == "__main__":
    main()
