"""Project schemas; project.json = {id, name, description, created_at} [doc 04 §1]."""

from pydantic import BaseModel, Field, model_validator


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def at_least_one_field(self) -> "ProjectUpdate":
        if self.name is None and self.description is None:
            raise ValueError("at least one of name or description is required")
        return self


class Project(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
