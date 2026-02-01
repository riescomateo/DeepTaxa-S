import glob
import os
import re
import sys

BASE_DIR = "/root/.cache/huggingface/modules/transformers_modules"

matches = glob.glob(
    os.path.join(BASE_DIR, "**", "flash_attn_triton.py"),
    recursive=True
)

if not matches:
    print("❌ flash_attn_triton.py not found.")
    print("➡️ Make sure the model has been loaded at least once.")
    sys.exit(1)

# If multiple matches exist, pick the newest one
file_path = max(matches, key=os.path.getmtime)

print(f"✅ Patching: {file_path}")

with open(file_path, "r") as f:
    content = f.read()

original_content = content

# ---- Patch 1: trans_b fix ----
content = content.replace(
    "tl.dot(q, k, trans_b=True)",
    "tl.dot(q, tl.trans(k))"
)
content = content.replace(
    "tl.dot(do, v, trans_b=True)",
    "tl.dot(do, tl.trans(v))"
)

# ---- Patch 2: memory config fix ----
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
    pattern = re.compile(
        r"triton\.Config\(\{\s*'BLOCK_M': 128,\s*'BLOCK_N': 128\s*\},\s*num_warps=8,\s*num_stages=1\),",
        re.DOTALL
    )
    content = pattern.sub(new_config, content)

# ---- Write back if needed ----
if content != original_content:
    with open(file_path, "w") as f:
        f.write(content)
    print("✅ File patched successfully.")
else:
    print("ℹ️ No changes needed (already patched).")
