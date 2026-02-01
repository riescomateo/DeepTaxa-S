import os
import re

file_path = "/root/.cache/huggingface/modules/transformers_modules/zhihan1996/DNABERT_hyphen_S/00e47f96cdea35e4b6f5df89e5419cbe47d490c6/flash_attn_triton.py"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

with open(file_path, 'r') as f:
    content = f.read()

original_content = content

# 1. Apply trans_b fix
# Replace tl.dot(q, k, trans_b=True) with tl.dot(q, tl.trans(k))
content = content.replace("tl.dot(q, k, trans_b=True)", "tl.dot(q, tl.trans(k))")
# Replace tl.dot(do, v, trans_b=True) with tl.dot(do, tl.trans(v))
content = content.replace("tl.dot(do, v, trans_b=True)", "tl.dot(do, tl.trans(v))")

# 2. Apply memory fix (BLOCK_M/N 128 -> 64)
old_config = """        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128
        },
                      num_warps=8,
                      num_stages=1),"""

new_config = """        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 64
        },
                      num_warps=4,
                      num_stages=1),"""

if old_config in content:
    content = content.replace(old_config, new_config)
else:
    # Regex fallback
    pattern = re.compile(r"triton\.Config\(\{\s*'BLOCK_M': 128,\s*'BLOCK_N': 128\s*\},\s*num_warps=8,\s*num_stages=1\),", re.DOTALL)
    if pattern.search(content):
        content = pattern.sub(new_config, content)

if content != original_content:
    with open(file_path, 'w') as f:
        f.write(content)
    print("File patched successfully (trans_b and memory fixes applied).")
else:
    print("No changes made. Patches might have been already applied or patterns not found.")
