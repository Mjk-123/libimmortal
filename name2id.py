# Print "NAME: id" in the exact order used by the encoder (enc.names)

from libimmortal.utils.aux_func import DEFAULT_ENCODER  # <-- fix import path if needed

enc = DEFAULT_ENCODER

for name in enc.names:
    idx = enc.name2id[name]
    print(f"{name}: {idx}")
