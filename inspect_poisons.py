import torch

path = "/Data/titouan.kerogues/data-taggants-runs/mnist-sign-smoke/poisons.pth"
obj = torch.load(path, map_location="cpu", weights_only=False)

for name in ["watermarks", "bounds", "data", "labels", "indices"]:
    value = obj[name]

    print(f"\n=== {name} ===")
    print("Type :", type(value))
    print("Nombre d'éléments :", len(value))

    if isinstance(value, dict):
        for key, item in list(value.items())[:3]:
            print("Clé :", key)
            print("Type :", type(item))
            print("Shape :", getattr(item, "shape", None))

            if isinstance(item, torch.Tensor):
                print("Min :", item.min().item())
                print("Max :", item.max().item())
    else:
        print("Exemple :", str(value[:3])[:300])