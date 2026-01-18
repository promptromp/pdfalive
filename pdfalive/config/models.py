"""Pydantic models for configuration file validation."""

from pydantic import BaseModel, ConfigDict, Field


class GlobalConfig(BaseModel):
    """Global configuration settings shared across all commands."""

    model_config = ConfigDict(populate_by_name=True)

    model_identifier: str | None = Field(default=None, alias="model-identifier")
    show_token_usage: bool | None = Field(default=None, alias="show-token-usage")


class GenerateTocConfig(BaseModel):
    """Configuration settings for the generate-toc command."""

    model_config = ConfigDict(populate_by_name=True)

    force: bool | None = None
    request_delay: float | None = Field(default=None, alias="request-delay")
    ocr: bool | None = None
    ocr_language: str | None = Field(default=None, alias="ocr-language")
    ocr_dpi: int | None = Field(default=None, alias="ocr-dpi")
    ocr_output: bool | None = Field(default=None, alias="ocr-output")
    postprocess: bool | None = None
    inplace: bool | None = None
    model_identifier: str | None = Field(default=None, alias="model-identifier")
    show_token_usage: bool | None = Field(default=None, alias="show-token-usage")


class ExtractTextConfig(BaseModel):
    """Configuration settings for the extract-text command."""

    model_config = ConfigDict(populate_by_name=True)

    ocr_language: str | None = Field(default=None, alias="ocr-language")
    ocr_dpi: int | None = Field(default=None, alias="ocr-dpi")
    force: bool | None = None
    inplace: bool | None = None


class RenameConfig(BaseModel):
    """Configuration settings for the rename command."""

    model_config = ConfigDict(populate_by_name=True)

    query: str | None = None
    yes: bool | None = None
    model_identifier: str | None = Field(default=None, alias="model-identifier")
    show_token_usage: bool | None = Field(default=None, alias="show-token-usage")


class PdfAliveConfig(BaseModel):
    """Root configuration model containing all command configurations."""

    model_config = ConfigDict(populate_by_name=True)

    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    generate_toc: GenerateTocConfig = Field(default_factory=GenerateTocConfig, alias="generate-toc")
    extract_text: ExtractTextConfig = Field(default_factory=ExtractTextConfig, alias="extract-text")
    rename: RenameConfig = Field(default_factory=RenameConfig)
