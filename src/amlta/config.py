from dataclasses import dataclass
from os import PathLike
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
