import os
from dataclasses import dataclass
from os import PathLike, environ
from pathlib import Path
from typing import Self


@dataclass
class Config:
    project_dir: Path
    data_dir: Path
    generated_dir: Path

    def __init__(
        self, project_dir: PathLike | None = None, data_dir: PathLike | None = None
    ):
        if project_dir is None:
            project_dir = Path(__file__).parent.parent.parent
        else:
            project_dir = Path(project_dir)

        if data_dir is None:
            data_dir = project_dir / "data"
        else:
            data_dir = Path(data_dir)

        generated_dir = data_dir / "generated"

        self.project_dir = project_dir
        self.data_dir = data_dir
        self.generated_dir = generated_dir

    def update(self, data_dir: PathLike | None = None) -> Self:
        new = type(self)(data_dir=data_dir)
        self.__dict__.update(new.__dict__)
        return self

    @property
    def ilcd_dir(self) -> Path:
        return self.data_dir / "ILCD"

    @property
    def ilcd_processes_xml_dir(self) -> Path:
        return self.ilcd_dir / "processes"

    @property
    def ilcd_processes_json_dir(self) -> Path:
        return self.ilcd_dir / "processes_json"


config = Config()

try:
    from google.colab import userdata  # pyright: ignore[reportMissingImports]  # noqa: F401, I001
except ImportError:
    IN_COLAB = False
else:
    IN_COLAB = True


if IN_COLAB:
    mount_point = Path("/content/drive")
    drive_path = mount_point / "MyDrive"

    data_dir = environ.get("COLAB_DATA_DIR")
    if not data_dir:
        try:
            data_dir = userdata.get("COLAB_DATA_DIR")

        except Exception:
            pass
        try:
            os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
        except Exception:
            pass

    if not data_dir:
        data_dir = drive_path / "uni" / "ws2425" / "amlta" / "project" / "data"

    data_dir = Path(data_dir)

    config.__init__(data_dir=data_dir)
