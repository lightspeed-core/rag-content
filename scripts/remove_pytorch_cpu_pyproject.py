from tomlkit import parse, dumps
from pathlib import Path

# This script removes the pytorch-cpu dependency from the pyproject.toml file.
# It is used to create a container image with GPU CUDA backend.


def remove_sections(file_path: str, sections_to_remove: list[str]):
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    doc = parse(content)

    for section in sections_to_remove:
        keys = section.split(".")
        current = doc
        for key in keys[:-1]:
            if key not in current:
                break
            current = current[key]
        else:
            current.pop(keys[-1], None)

    path.write_text(dumps(doc), encoding="utf-8")


if __name__ == "__main__":
    file_path = "pyproject.toml"
    print(f"pyproject file path: {file_path}")
    sections = ["tool.uv.index", "tool.uv.sources"]
    remove_sections(file_path, sections)
