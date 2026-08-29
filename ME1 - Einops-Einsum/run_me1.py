"""Execute the ME1 notebook in place and report the test accuracy."""
import json
import nbformat
from nbclient import NotebookClient

NB = "ME1_einops_einsum_cnn.ipynb"
nb = nbformat.read(NB, as_version=4)
client = NotebookClient(nb, timeout=3600, kernel_name="ai231",
                        resources={"metadata": {"path": "."}})
client.execute()

# sanity: the run must have used CUDA, not a CPU fallback
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    for out in cell.get("outputs", []):
        text = out.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        if "device:" in text:
            print(text.strip())
            assert "cuda" in text, "RAN ON CPU — expected CUDA!"
nbformat.write(nb, NB)

# extract the test accuracy from outputs
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    for out in cell.get("outputs", []):
        text = out.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        if "TEST ACCURACY" in text:
            print(text.strip())
